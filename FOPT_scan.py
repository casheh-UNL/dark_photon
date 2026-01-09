# Calculates relevant data for a given FOPT.

import numpy as np
import sys
import os
import time
import json
import signal
from contextlib import contextmanager

from kinetic_mixing import KineticMixing


class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutException after specified seconds."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Computation exceeded {seconds} second time limit")
    
    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


# Find the absolute path to the ELENA/src directory
src_path = os.path.abspath('ELENA/src')

# Add this path to the start of sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from temperatures import find_T_min, find_T_max, refine_Tmin, R_sepH
from espinosa import Vt_vec
from utils import interpolation_narrow
from temperatures import compute_logP_f, N_bubblesH, R_sepH, R0, compute_Gamma_f
from GWparams import cs2, alpha_th_bar, beta, GW_SuperCooled
from model_plus_fermion import model_plus_fermion
# from dof_interpolation import g_rho_splin



def lambda_max(g, y):
    '''Condition for maintaining potential barrier at T=0.'''
    c = 5 + np.log(4)
    g4, y4 = g**4, y**4

    if not ((32*np.pi**4 / c) > 48*g4 - y4):
        raise ValueError("The given g and y values yield a complex-valued λ_max.")
    
    # using the negative root
    if not (48*g4 - y4 > 0):
        raise ValueError("Quartic coupling λ_max is negative.")
    return (8*np.pi**2 - np.sqrt(64*np.pi**4 - 2*c*(48*g4 - y4))) / (2. * c)


def sample_parameters(g_min=1e-2, g_max=None, y_min=1e-2, y_max=None, eps=0.1):
    """Sample random parameters g and y that satisfy the required constraints."""
    if g_max is None:
        g_max = np.sqrt(4*np.pi)
    if y_max is None:
        y_max = np.sqrt(4*np.pi)
    
    found = False
    while not found:
        g_scan = 10**(np.random.uniform(np.log10(g_min), np.log10(g_max)))
        y_scan = 10**(np.random.uniform(np.log10(y_min), np.log10(y_max)))
        # lambda_scan = 10**(np.random.uniform(np.log10(1e-4), np.log10(4*np.pi)))
        # found = True

        g4, y4 = g_scan**4, y_scan**4
        b = 32*np.pi**4 / (5 + np.log(4))
        if 0 < 48*g4 - y4 < b:
            # lambda_max will be real and positive
            lambda_scan = lambda_max(g_scan, y_scan) * (1 - np.random.uniform(0, eps))
            if lambda_scan < 2:  # ELENA cannot handle values that are too high
                found = True
    
    return g_scan, y_scan, lambda_scan


def is_increasing(arr):
    """Check if array is monotonically increasing."""
    return np.all(arr[:-1] <= arr[1:])

# --- Masses ---
def m2_gaugeboson(g, v): return g**2 * v**2 * 4  # last factor of 2^2 from scalar's charge
def m2_scalar(lam, v): return 2 * lam * v**2     # symmetry-breaking scalar
def m2_fermion(y, v): return (y**2 / 2.) * v**2





def compute_fopt_point(g, y, lam=None, vev=1.0, units='GeV', v_w=1.0, n_points=100, verbose=False):
    """
    Compute all FOPT parameters for a given point in parameter space.
    
    Parameters:
    -----------
    g : float
        Dark gauge coupling
    y : float
        Yukawa coupling
    vev : float
        Vacuum expectation value
    units : str
        Units for temperature
    v_w : float
        Domain wall speed
    n_points : int
        Number of temperature points for sampling
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    dict : Dictionary containing all computed parameters, or None if computation fails
    """
    
    if verbose:
        print(f"Computing point: g={g:.4e}, y={y:.4e}, lambda={lam:.4e}")
    
    try:
        # Initialize point dictionary
        point = {}
        point['VEV'] = vev
        point['v_w'] = v_w
        point['lambda'] = lambda_max(g, y) if lam is None else lam
        point['gPrime'] = g
        point['y'] = y
        
        lambda_ = point['lambda']

        point['m_scalar'] = np.sqrt(m2_scalar(lambda_, vev))
        point['m_gaugeboson'] = np.sqrt(m2_gaugeboson(g, vev))
        point['m_fermion'] = np.sqrt(m2_fermion(y, vev))
        
        # Initialize model
        model = model_plus_fermion(vev, lambda_, g, y, xstep=vev*1e-3, Tstep=vev*1e-3)
        V = model.DVtot
        dV = model.gradV
        
        # Bounding Temperatures
        T_max, vevs_max, max_min_vals, false_min_tmax = find_T_max(
            V, dV, precision=1e-2, Phimax=2*vev, step_phi=vev*1e-2, tmax=2.5*vev
        )
        T_min, vevs_min, false_min_tmin = find_T_min(
            V, dV, tmax=T_max, precision=1e-2, Phimax=2*vev, 
            step_phi=vev*1e-2, max_min_vals=max_min_vals
        )
        
        if T_max is not None and T_min is not None:
            maxvev = np.max(np.concatenate((vevs_max, vevs_min)))
        elif T_max is not None:
            maxvev = np.max(vevs_max)
        elif T_min is not None:
            maxvev = np.max(vevs_min)
        else:
            maxvev = None
        
        T_min = refine_Tmin(T_min, V, dV, maxvev, log_10_precision=6) if T_min is not None else None
        
        point['T_max'] = T_max
        point['T_min'] = T_min
        
        # Euclidean Action - using dictionaries as closure variables
        true_vev = {}
        S3overT = {}
        V_min_value = {}
        phi0_min = {}
        V_exit = {}
        false_vev = {}
        
        def action_over_T(T, c_step_phi=1e-3, precision=1e-3):
            instance = Vt_vec(T, V, dV, step_phi=c_step_phi, precision=precision, 
                            vev0=maxvev, ratio_vev_step0=50)
            if instance.barrier:
                true_vev[T] = instance.true_min
                false_vev[T] = instance.phi_original_false_vev
                S3overT[T] = instance.action_over_T
                V_min_value[T] = instance.min_V
                phi0_min[T] = instance.phi0_min
                V_exit[T] = instance.V_exit
                return instance.action_over_T
            else:
                return None
        
        temperatures = np.linspace(T_min, T_max, n_points)
        action_vec = np.vectorize(action_over_T)
        action_vec(temperatures)
        temperatures = np.array([T for T in temperatures if T in S3overT])
        
        # Nucleation, Percolation, Completion
        is_physical = True
        
        counter = 0
        while counter <= 1:
            if counter == 1:
                temperatures = np.linspace(
                    np.nanmax([T_min, 0.95 * T_completion]), 
                    np.nanmin([T_max, 1.05 * T_nuc]), 
                    n_points, endpoint=True
                )
                action_vec(temperatures)
            
            logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(
                model, V_min_value, S3overT, v_w, units=units, cum_method='None'
            )
            Gamma_f_list, Temps_G, ratio_V, Gamma_list, H = compute_Gamma_f(
                model, V_min_value, S3overT, v_w, logP_f, units='GeV', 
                cum_method='cumulative_simpson'
            )
            
            RH, R = R_sepH(Temps, Gamma, logP_f, H, ratio_V)
            nH = N_bubblesH(Temps, Gamma, logP_f, H, ratio_V)
            
            mask_nH = ~np.isnan(nH)
            T_nuc = interpolation_narrow(np.log(nH[mask_nH]), Temps[mask_nH], 0)
            mask_Pf = ~np.isnan(logP_f)
            T_perc = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.71))
            T_completion = interpolation_narrow(logP_f[mask_Pf], Temps[mask_Pf], np.log(0.01))
            idx_compl = np.max([np.argmin(np.abs(Temps - T_completion)), 1])
            test_completion = np.array([logP_f[idx_compl - 1], logP_f[idx_compl], logP_f[idx_compl + 1]])
            test_completion = test_completion[~np.isnan(test_completion)]
            
            # if verbose:
            #     print(counter, T_completion, test_completion)
            #     print(is_increasing(test_completion))
            
            if not is_increasing(test_completion):
                T_completion = np.nan
            
            if counter == 1:
                d_dT_logP_f = np.gradient(logP_f, Temps)
                log_at_T_perc = interpolation_narrow(Temps, d_dT_logP_f, T_perc)
                ratio_V_at_T_perc = interpolation_narrow(Temps, ratio_V, T_perc)
                log_at_T_completion = interpolation_narrow(Temps, d_dT_logP_f, T_completion)
                ratio_V_at_T_completion = interpolation_narrow(Temps, ratio_V, T_completion)
                if ratio_V_at_T_perc > log_at_T_perc:
                    is_physical = False
                    if verbose:
                        print("\n *** The physical volume at percolation is not decreasing. The production of GW is questionable ***")
            counter += 1
        
        milestones = [T_max, T_nuc, T_perc, T_completion, T_min]
        milestones = [milestone for milestone in milestones if milestone is not None and not np.isnan(milestone)]
        action_vec(milestones)
        
        point['T_nuc'] = T_nuc
        point['T_perc'] = T_perc
        point['T_completion'] = T_completion
        
        # Check if T_perc is valid before continuing
        if T_perc is None or np.isnan(T_perc):
            if verbose:
                print("T_perc is None or NaN - skipping point")
            return None
        
        # Gravitational Wave Parameters
        if T_perc is not None:
            action_over_T(T_perc)
            c_s2 = cs2(T_perc, model, true_vev, units=units)[0]
        
        def c_alpha_inf(T, units):
            '''
            "Efficiency parameter" representing the minimal α required to overcome leading-order
            friction in plasma due to changes in field masses.
            '''
            v_true = true_vev[T]
            v_false = false_vev[T]
            Dm2_photon = 1*3 * (m2_gaugeboson(g, v_true) - m2_gaugeboson(g, v_false))
            Dm2_Phi = 1*1 * (m2_scalar(lambda_, v_true) - m2_scalar(lambda_, v_false)) # original ELENA multiplied this by 3/2
            Dm2_DM = 0.5*4 * (m2_fermion(g, v_true) - m2_fermion(g, v_false))
            numerator = (Dm2_photon + Dm2_Phi + Dm2_DM) * T**2 / 24
            rho_tot = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM=True, units=units)) / 4
            rho_DS = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM=False, units=units)) / 4
            return numerator/ rho_tot, numerator / rho_DS
        
        def c_alpha_eq(T, units):
            '''
            If domain walls reach relativistic speeds, next-to-leading-order friction terms
            (proportional to the Lorentz factor) become relevant. This α_eq is used in calculating
            the terminal speed of walls.
            '''
            v_true = true_vev[T]
            v_false = false_vev[T]
            numerator = (g**2 * 3 * (g * (v_true - v_false)) * T**3)
            rho_tot = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM=True, units=units)) / 4
            rho_DS = - T * 3 * (model.dVdT(v_false, T, include_radiation=True, include_SM=False, units=units)) / 4
            return numerator / rho_tot, numerator / rho_DS
        
        alpha, alpha_DS = alpha_th_bar(T_perc, model, V_min_value, false_vev, true_vev, units=units)
        alpha_inf, alpha_inf_DS = c_alpha_inf(T_perc, units)
        alpha_eq, alpha_eq_DS = c_alpha_eq(T_perc, units)
        
        alpha = alpha[0]
        
        gamma_eq = (alpha - alpha_inf) / alpha_eq
        
        if alpha < alpha_inf:
            is_physical = False
            if verbose:
                print("\n*** Warning, the bubble expansion is not in runaway regime! The results of the computation are not reliable. ***")
        
        v_min = 0.99
        if gamma_eq < 1 / np.sqrt(1 - v_min**2):
            is_physical = False
            if verbose:
                print(f"\n*** Warning, the NLO pressure could prevent the walls to reach relativistic velocities (gamma_eq = {gamma_eq}). The results of the computation are not reliable as v_w = {v_w} was used. ***")
        
        # Mean Bubble Separation
        RH, R = R_sepH(Temps, Gamma, logP_f, H, ratio_V)
        RH_interp = interpolation_narrow(Temps, RH, T_perc)
        H_star = interpolation_narrow(Temps, H, T_perc)
        R_star = RH_interp / H_star
        gamma_star = 2 * R_star / (3 * R0(T_perc, S3overT, V_exit))
        
        # GW spectrum peak
        dark_dof = 2*2 + 1*3 + 1*1
        GW = GW_SuperCooled(T_perc, alpha, alpha_inf, alpha_eq, R_star, gamma_star, 
                           H_star, np.sqrt(c_s2), v_w, units, dark_dof)
        GW_peaks = GW.find_peak(verbose=verbose)
        kappa_col = GW.kappa_col
        tau_sw = GW.tau_sw
        T_reh = GW.T_reh
        
        # β and γ
        logP_f, Temps, ratio_V, Gamma, H = compute_logP_f(
            model, V_min_value, S3overT, v_w, units=units, cum_method='None'
        )
        beta_Hn, gamma_Hn, times, Gamma_t, Temps_t, H_t = beta(
            Temps, ratio_V, Gamma, H, T_nuc, T_perc, verbose=verbose
        )
        
        # Final updates to point
        point['is_physical'] = is_physical
        point['c_s2'] = c_s2
        point['alpha'] = alpha
        point['alpha_inf'] = alpha_inf
        point['alpha_eq'] = alpha_eq
        point['R_sepH_perc'] = RH_interp
        point['H_perc'] = H_star
        point['gamma_star'] = gamma_star
        point['GW_peak'] = GW_peaks[0]
        point['GW_peak_col'] = GW_peaks[1]
        point['GW_peak_sw'] = GW_peaks[2]
        point['GW_peak_turb'] = GW_peaks[3]
        point['kappa_col'] = kappa_col
        point['tau_sw'] = tau_sw
        point['T_reh'] = T_reh
        point['beta_over_H'] = beta_Hn
        point['gamma_over_H'] = gamma_Hn

        # --- Kinetic Mixing ---
        KM = KineticMixing(gPrime=g, m_DM=point['m_fermion'], m_Zprime=point['m_gaugeboson'],
                           QPrime_DM=(1/2, -1/2), QPrime_SM=(0, 0))
        # DM charges are chosen for an axial interaction, and the 1/2 factor comes from the effective
        # Lagrangian interaction of two-component DM (see p. 7 of https://arxiv.org/pdf/2502.19478)
    
        epsilon_solution = KM.solve_kinetic_mixing(target_sigma_v=2.5e-10/0.12, v=np.sqrt(6/20))
        # using v^2 ~ 6/x for cold relics and with x := m/T = 20
        if epsilon_solution.converged:
            epsilon = epsilon_solution.root

            gZ_DM, gZp_DM = KM.Z_Zprime_couplings(kinetic_mixing=epsilon, Q_em=0, T3_L=0, T3_R=0, QPrime_LR=(1,1))
            DM_coupling_ratio = gZ_DM / gZp_DM
            # This compares DM coupling to Z and Z'. If this ratio is large, we need to add Z contributions.

            GZ, GZp = np.array([0, 0]), np.array([0, 0])
            for f in KM.final_states:
                Q, T3L, T3R, = KM.get_SM_charges(f)
                gZ, gZp = KM.Z_Zprime_couplings(epsilon, Q, T3L, T3R, (0, 0))
                GZ = GZ + abs(gZ)
                GZp = GZp + abs(gZp)
            GZ = GZ / len(KM.final_states)
            GZp = GZp / len(KM.final_states)
            SM_coupling_ratio = GZp / GZ
            # This compares SM coupling to Z and Z'. If this ratio is large, we need to add Z contributions.
        else:
            epsilon = np.nan
            DM_coupling_ratio, SM_coupling_ratio = [np.nan, np.nan], [np.nan, np.nan]

        point['kinetic_mixing'] = epsilon
        point['DM_coupling_ratio'] = DM_coupling_ratio
        point['SM_coupling_ratio'] = SM_coupling_ratio
        
        # Convert numpy types to Python native types for JSON serialization
        point = convert_to_serializable(point)
        
        return point
        
    except Exception as e:
        import traceback
        if verbose:
            print(f"Error computing point: {e}")
            traceback.print_exc()
        return None


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        # Check for NaN before converting
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif obj is None:
        return None
    # Handle regular Python float that might be NaN
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj


def compute_multiple_points(n_points=100, output_file='FOPT_parameters_.json', 
                           g_min=1e-2, g_max=None, y_min=1e-2, y_max=None, eps=0.1,
                           vev=1.0, units='GeV', v_w=1.0, timeout=60, verbose=True):
    """
    Compute multiple FOPT points and save to JSON file.
    
    Parameters:
    -----------
    n_points : int
        Number of successful points to compute
    output_file : str
        Output JSON filename
    g_min, g_max, y_min, y_max : float
        Parameter ranges for sampling
    vev, units, v_w : float/str
        Physical parameters
    verbose : bool
        Print progress
    """
    
    results = []
    successful_points = 0
    total_attempts = 0
    
    print(f"Computing {n_points} successful FOPT points...")
    print(f"Results will be saved to {output_file}")
    
    start_time = time.time()
    
    while successful_points < n_points:
        point_start = time.time()
        total_attempts += 1
        
        # Sample parameters
        g, y, lam = sample_parameters(g_min, g_max, y_min, y_max, eps)
        
        # Compute point with timeout
        try:
            with time_limit(timeout):
                point = compute_fopt_point(g, y, lam, vev=vev, units=units, v_w=v_w, 
                                        n_points=100, verbose=verbose)
        except TimeoutException as e:
            if verbose:
                print(f"Attempt {total_attempts} timed out after 60 seconds (successful: {successful_points}/{n_points})")
            point = None
        
        if point is not None:
            results.append(point)
            successful_points += 1
            
            point_time = time.time() - point_start
            elapsed = time.time() - start_time
            avg_time = elapsed / successful_points
            eta = avg_time * (n_points - successful_points)
            
            if verbose:
                print(f"Point {successful_points}/{n_points} completed in {point_time:.2f}s "
                    f"(Attempts: {total_attempts}, Success rate: {successful_points/total_attempts*100:.1f}%, "
                    f"ETA: {eta/60:.1f}m)")
            
            # Save result immediately after each successful point
            with open(output_file, 'a') as f:
                f.write(json.dumps(point) + '\n')
        else:
            if verbose:
                print(f"Attempt {total_attempts} failed (successful: {successful_points}/{n_points})")
    
    total_time = time.time() - start_time
    
    print(f"\nCompleted!")
    print(f"Successful points: {successful_points}")
    print(f"Total attempts: {total_attempts}")
    print(f"Success rate: {successful_points/total_attempts*100:.1f}%")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per successful point: {total_time/successful_points:.2f} seconds")
    print(f"Results saved to {output_file}")
    
    return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute FOPT parameters for multiple points')
    parser.add_argument('-n', '--n_points', type=int, default=1,
                       help='Number of points to compute (default: 1)')
    parser.add_argument('--vev', type=float, default=1.0,
                       help='Vacuum expectation value in GeV (default: 1.0)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output JSON file (default: dark_photon/FOPT_data/FOPT_parameters_{vev}.json)')
    parser.add_argument('--g_min', type=float, default=1e-2,
                       help='Minimum g value (default: 1e-2)')
    parser.add_argument('--g_max', type=float, default=None,
                       help='Maximum g value (default: sqrt(4*pi))')
    parser.add_argument('--y_min', type=float, default=1e-2,
                       help='Minimum y value (default: 1e-2)')
    parser.add_argument('--y_max', type=float, default=None,
                       help='Maximum y value (default: sqrt(4*pi))')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--timeout', type=int, default=60,
                   help='Timeout in seconds for each point computation (default: 60)')
    parser.add_argument('--test', nargs='+', type=float,
                       help='Test a specific (g, y) or (g, y, lam) point instead of scanning')
    
    args = parser.parse_args()
    
    # Set default output file based on VEV if not specified
    if args.output is None:
        args.output = f'dark_photon/FOPT_data/FOPT_parameters_{args.vev}.json'
    
    if args.test:
        # Test mode: compute a single specific point
        if len(args.test) == 2:
            g, y = args.test
            lam = None
        elif len(args.test) == 3:
            g, y, lam = args.test
        else:
            print("Error: --test requires 2 or 3 arguments (g, y) or (g, y, lam)")
            return
        
        lam_str = f", lambda={lam}" if lam is not None else ""
        print(f"Testing point: g={g}, y={y}{lam_str}, VEV={args.vev}")
        result = compute_fopt_point(g, y, vev=args.vev, verbose=True, lam=lam)
        if result:
            print("\nSuccess! Point data:")
            print(json.dumps(result, indent=2))
            # Append to file - one JSON object per line
            with open(args.output, 'a') as f:
                f.write(json.dumps(result) + '\n')
            print(f"\nAppended to {args.output}")
        else:
            print("\nPoint computation failed.")
    else:
        # Normal mode: compute multiple points
        compute_multiple_points(
            n_points=args.n_points,
            output_file=args.output,
            g_min=args.g_min,
            g_max=args.g_max,
            y_min=args.y_min,
            y_max=args.y_max,
            vev=args.vev,
            timeout=args.timeout,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()
