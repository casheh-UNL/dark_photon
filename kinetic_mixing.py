"""
This class determines annihilation cross sections for
DM + DM -> Z' -> f fbar
summed over all SM fermions.
"""

import numpy as np
from particle import Particle
from particle.pdgid import is_quark, is_lepton
from scipy.optimize import root_scalar

# --- Electroweak inputs ---
sW2 = 0.23121
cW2 = 1 - sW2
M_Z, Higgs_VEV = 91.1876, 246.0
g_Z = 2 * M_Z / Higgs_VEV

sW, cW = np.sqrt(sW2), np.sqrt(cW2)
g1, g2 = g_Z * sW, g_Z * cW

# --- Helper functions ---
def decay_rate(gL, gR, m_f, m_V, symm_factor=1):
    '''
    Decay rate (width) for a vector
    boson decaying to di-fermions.
    
    Parameters
    ----------
    QL, QR : floats
        Couplings between vector boson and left-, right-handed spinor
    m_f : float
        Fermion mass
    m_V : float
        Vector boson mass
    symm_factor : float
        Optional symmetry factor when final state particles are identical.
    '''
    if 2 * m_f >= m_V:
        return 0.0
    
    beta = np.sqrt(1 - 4*m_f**2 / m_V**2)
    return m_V / (24*np.pi) * ((gL**2 + gR**2) * (1 - m_f**2 / m_V**2) + 6*gL*gR * m_f**2 / m_V**2) * beta / symm_factor
    

class KineticMixing(object):
    def __init__(self, gPrime, m_DM, m_Zprime,
             QPrime_DM=(1.0, 1.0),
             QPrime_SM=(0.0, 0.0)):
        """
        Parameters
        ----------
        gPrime : float
            U(1)' gauge coupling
        m_DM : float
            Dark matter mass
        m_Zprime : float
            Z' mass
        QPrime_DM = (Q_L, Q_R) for DM
        QPrime_SM = (Q_L, Q_R) for SM fermions
        """
        self.gPrime = gPrime
        self.m_DM = m_DM
        self.m_Zprime = m_Zprime

        self.QPrime_DM = np.asarray(QPrime_DM, dtype=float)
        self.QPrime_SM = np.asarray(QPrime_SM, dtype=float)

        self.final_states = self.SM_fermions()


    # ------------------------------------------------------------------
    # Final state utilities
    # ------------------------------------------------------------------

    def SM_fermions(self):
        """All kinematically accessible SM fermions (no antifermions)."""
        return [
            p for p in Particle.all()
            if (
                (is_quark(p.pdgid) or is_lepton(p.pdgid))  # Check if quark or lepton
                and p.pdgid > 0  # Only particles, not antiparticles
                and p.mass is not None
                and p.mass / 1e3 < self.m_DM # Convert MeV to GeV
                and abs(p.pdgid) in [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]  # SM fermions only
            )
        ]

    def color_factor(self, particle):
        """Nc = 3 for quarks, 1 otherwise."""
        return 3 if is_quark(particle.pdgid) else 1
    
    def get_SM_charges(self, particle):
        """
        Extract Standard Model charges from a Particle object.
        
        Parameters
        ----------
        particle : Particle
            Particle object from Scikit-HEP
        
        Returns
        -------
        tuple
            (Q_em, T3_L, T3_R) - electric charge, left- and right-handed weak isospin
        """
        
        # Electric charge (in units of e)
        Q_em = particle.charge
        
        pdg = abs(particle.pdgid)
        
        # Weak isospin T3_L
        # Left-handed fermions are in SU(2)_L doublets
        if is_quark(pdg):
            # Up-type quarks (u, c, t): T3_L = +1/2
            # Down-type quarks (d, s, b): T3_L = -1/2
            if pdg in [2, 4, 6]:  # up, charm, top
                T3_L = 1/2
            else:  # down, strange, bottom
                T3_L = -1/2
        elif is_lepton(pdg):
            # Neutrinos: T3_L = +1/2
            # Charged leptons: T3_L = -1/2
            if pdg in [12, 14, 16]:  # neutrinos
                T3_L = 1/2
            else:  # electron, muon, tau
                T3_L = -1/2
        else:
            T3_L = 0.0
        
        # Right-handed fermions are SU(2)_L singlets
        T3_R = 0.0
        
        return Q_em, T3_L, T3_R


    # ------------------------------------------------------------------
    # Z–Z' kinetic mixing charges
    # ------------------------------------------------------------------

    def Z_Zprime_mixing(self, kinetic_mixing):
        '''
        Compute mass mixing angle between Z and Z'.

        Parameters
        ----------
        kinetic_mixing : float
            Kinetic mixing angle
        '''
        epsilon = kinetic_mixing
        tan_eps = np.tan(epsilon)

        M_Z2, M_Zp2 = M_Z**2, self.m_Zprime**2

        # Lagrangian mass parameters
        disc = np.sqrt((M_Z2 - M_Zp2)**2 - 4 * sW2 * tan_eps**2 * M_Z2 * M_Zp2)

        m_Z2  = (M_Z2 + M_Zp2 + disc) / (2 + 2*sW2 * tan_eps**2)
        m_Zp2 = (M_Z2 + M_Zp2 - disc) / (2 / np.cos(epsilon)**2)
        # notice as kinetic mixing goes to zero, m_Z2 --> M_Z2 and similarly for Zp

        # Mass matrix elements
        M2_11 = m_Z2
        M2_12 = m_Z2 * sW * tan_eps
        M2_22 = m_Z2 * sW**2 * tan_eps**2 + (m_Zp2 / np.cos(epsilon))**2

        return np.arctan(2*M2_12 / (M2_11 - M2_22)) / 2

    def Z_Zprime_couplings(self, kinetic_mixing, Q_em, T3_L, T3_R=0.0, QPrime_LR=(1.0, 1.0)):
        """
        Compute left/right couplings of a fermion under Z and Z'
        after kinetic mixing.

        Parameters
        ----------
        kinetic_mixing : float
            Kinetic mixing angle epsilon
        Q_em : float
            Electric charge
        T3_L : float
            Weak isospin (left-handed)
        T3_R : float, optional
            Weak isospin (right-handed), default 0
        QPrime_LR : array-like
            (Q'_L, Q'_R).

        Returns
        -------
        (gZ_L, gZ_R), (gZp_L, gZp_R)
        """

        epsilon = kinetic_mixing
        xi = self.Z_Zprime_mixing(kinetic_mixing)
        gPrime = self.gPrime

        # Electroweak charges
        QZ_LR = np.array([T3_L, T3_R]) - sW2 * Q_em

        # U(1)' charges
        QPrime_LR = np.asarray(QPrime_LR, dtype=float)

        # Z couplings
        gZ_LR = (
            g1*cW * (-cW * np.sin(xi) * np.tan(epsilon)) * Q_em
            + g2/cW * (np.cos(xi) + sW * np.sin(xi) * np.tan(epsilon)) * QZ_LR
            + gPrime * (np.sin(xi) / np.cos(epsilon)) * QPrime_LR
        )

        # Z' couplings
        gZprime_LR = (
            g1*cW * (-cW * np.cos(xi) * np.tan(epsilon)) * Q_em
            + g2/cW * (-np.sin(xi) + sW * np.cos(xi) * np.tan(epsilon)) * QZ_LR
            + gPrime * (np.cos(xi) / np.cos(epsilon)) * QPrime_LR
        )

        return gZ_LR, gZprime_LR
    

    # ------------------------------------------------------------------
    # Z' total width
    # ------------------------------------------------------------------

    def mediator_decay_rate(self, kinetic_mixing):
        """
        Total Z' decay rate (width) from:
        Z' -> f fbar (SM fermions)
        Z' -> chi chibar (DM)
        with chiral couplings.

        Parameters
        ----------
        kinetic_mixing : float
            Kinetic mixing angle epsilon
        """
        mV = self.m_Zprime
        Gamma = 0.0

        # --- SM fermions ---
        for f in self.final_states:
            mf = f.mass / 1e3  # Convert MeV to GeV
            Nc = self.color_factor(f)
            
            # Get SM charges
            Q_em, T3_L, T3_R = self.get_SM_charges(f)
            
            # Get kinetic mixing-dependent couplings
            _, (gZp_L, gZp_R) = self.Z_Zprime_couplings(
                kinetic_mixing, Q_em, T3_L, T3_R, self.QPrime_SM
            )
            
            Gamma_f = Nc * decay_rate(gZp_L, gZp_R, mf, mV, symm_factor=1)
            Gamma += Gamma_f

        # --- DM (SM singlet: Q_em=0, T3_L=0, T3_R=0) ---
        mX = self.m_DM
        
        # Get kinetic mixing-dependent DM couplings
        _, (gZp_L_DM, gZp_R_DM) = self.Z_Zprime_couplings(
            kinetic_mixing, Q_em=0.0, T3_L=0.0, T3_R=0.0, QPrime_LR=self.QPrime_DM
        )
        
        Gamma_DM = decay_rate(gZp_L_DM, gZp_R_DM, mX, mV, symm_factor=1)
        Gamma += Gamma_DM

        return Gamma

    # ------------------------------------------------------------------
    # Annihilation cross section
    # ------------------------------------------------------------------

    def sigma_ann(self, s, m_f, g_DM, g_f, symm_factor=1):
        '''
        Annihilation cross section of DM DM -> Z' -> f f̄

        Parameters
            ----------
            s : float
                Mandelstam invariant
            m_f : float
                Final-state fermion mass
            g_DM : array-like of shape (2,)
                Left- and right- handed DM couplings
            g_f : array-like of shape (2,)
                Left- and right- handed final-state couplings
            symm_factor : int
                Symmetry factor for identical particles
        '''
        # Kinematic thresholds
        if 4*m_f**2 >= s:
            return 0.0
        
        if 4*self.m_DM**2 >= s:
            return 0.0

        # Useful coupling combinations
        gDM_L, gDM_R = g_DM
        gf_L, gf_R = g_f

        gDM2 = gDM_L**2 + gDM_R**2
        gf2 = gf_L**2 + gf_R**2

        gDM_LR = gDM_L * gDM_R
        gf_LR = gf_L * gf_R
        
        # --- Masses ---
        m_DM = self.m_DM
        m_Zp = self.m_Zprime
        
        # --- Kinematics ---
        beta_f = np.sqrt(1.0 - 4*m_f**2 / s)
        beta_DM = np.sqrt(1.0 - 4*m_DM**2 / s)

        # --- Cross section ---
        coeff = (1/symm_factor) * (s / (48*np.pi * (s - m_Zp**2)**2)) * beta_f / beta_DM

        term1 = (1 - m_f**2/s) * (gDM2) * (gf2)
        term2 = (6*m_f**2 / s) * gf_LR * (gDM2 - (2*m_DM**2 / s) * (gDM2 - 4*gDM_LR))
        term3 = - (m_DM**2 / s) * (gf2) * (gDM2 - 6*gDM_LR - (4*m_f**2 / s) * (gDM2 - 3*gDM_LR))

        return coeff * (term1 + term2 + term3)

    # ------------------------------------------------------------------
    # "Thermally averaged" annihilation cross section
    # ------------------------------------------------------------------
    
    def sigma_v_exact(self, m_f, g_DM, g_f, v=0.0, symm_factor=1):
        """
        σv for DM DM -> Z' -> f f̄
        Calculated using the exact expression s = 2m_DM^2 (1 + γ).
        
        Parameters
        ----------
        m_f : float
            Final-state fermion mass
        g_DM : array-like of shape (2,)
            Left- and right- handed DM couplings
        g_f : array-like of shape (2,)
            Left- and right- handed final-state couplings
        v : float
            Relative speed
        symm_factor : int
            Symmetry factor for identical particles
        """
        # # Kinematic threshold
        # if m_f >= self.m_DM:
        #     return 0.0
        s = 2 * self.m_DM**2 * (1 + 1/np.sqrt(1 - v**2))
        sigma = self.sigma_ann(s, m_f, g_DM, g_f, symm_factor)

        return sigma * v
    
    def sigma_v_approx(self, m_f, kinetic_mixing=0.0, v=0.0, Q_em_f=None, T3_L_f=None, T3_R_f=0.0, symm_factor=1):
        """
        NR expansion of <σ v> for DM DM -> Z' -> f f̄
        Expanded to O(v^2) using s ≈ 4m^2 [1 + (v/2)^2].
        
        Parameters
        ----------
        m_f : float
            Final-state fermion mass
        kinetic_mixing : float
            Kinetic mixing angle epsilon
        v : float
            Relative velocity
        Q_em_f : float
            Electric charge of final state fermion
        T3_L_f : float
            Left-handed weak isospin of final state fermion
        T3_R_f : float
            Right-handed weak isospin (default 0)
        symm_factor : int
            Symmetry factor for identical particles
        """
        # Kinematic threshold
        if m_f >= self.m_DM:
            return 0.0
        
        # --- Couplings from kinetic mixing ---
        # DM couplings (SM singlet)
        _, (gDM_L, gDM_R) = self.Z_Zprime_couplings(
            kinetic_mixing, Q_em=0.0, T3_L=0.0, T3_R=0.0, QPrime_LR=self.QPrime_DM
        )
        
        # Final state fermion couplings
        _, (gf_L, gf_R) = self.Z_Zprime_couplings(
            kinetic_mixing, Q_em_f, T3_L_f, T3_R_f, QPrime_LR=self.QPrime_SM
        )
        
        # Useful coupling combinations
        gDM_L2 = gDM_L**2
        gDM_R2 = gDM_R**2
        gf_L2 = gf_L**2
        gf_R2 = gf_R**2
        gDM_LR = gDM_L * gDM_R
        gf_LR = gf_L * gf_R
        
        # --- Masses ---
        m_DM = self.m_DM
        m_Zp = self.m_Zprime
        
        # --- Kinematics ---
        beta = np.sqrt(1.0 - m_f**2 / m_DM**2)
        
        # --- s-wave (v^0) coefficient ---
        a_num = (
            gf_LR * gDM_L2 * m_f**2
            - (gf_L2 - 4*gf_LR + gf_R2) * gDM_LR * m_f**2
            + gf_LR * gDM_R2 * m_f**2
            + (gf_L2 + gf_R2) * (gDM_L + gDM_R)**2 * m_DM**2
        )
        
        a = beta * a_num / (8.0 * np.pi * symm_factor * (-4*m_DM**2 + m_Zp**2)**2)
        
        # --- p-wave (v^2) coefficient ---
        b_num = (
            3*gf_LR * m_f**2 * (
                -8*(gDM_L2 + 12*gDM_LR + gDM_R2) * m_f**2 * m_DM**2
                + 4*(gDM_L2 + 20*gDM_LR + gDM_R2) * m_DM**4
                + (-2*(gDM_L2 - 4*gDM_LR + gDM_R2)*m_f**2 
                + (3*gDM_L2 - 4*gDM_LR + 3*gDM_R2)*m_DM**2) * m_Zp**2
            )
            + gf_L2 * (
                -8*(gDM_L2 - 9*gDM_LR + gDM_R2) * m_f**4 * m_DM**2
                - 4*(2*gDM_L2 + 39*gDM_LR + 2*gDM_R2) * m_f**2 * m_DM**4
                + 4*(gDM_L2 + 18*gDM_LR + gDM_R2) * m_DM**6
                + (2*(gDM_L2 - 3*gDM_LR + gDM_R2)*m_f**4 
                + (-10*gDM_L2 + 3*gDM_LR - 10*gDM_R2)*m_f**2*m_DM**2 
                + (11*gDM_L2 + 6*gDM_LR + 11*gDM_R2)*m_DM**4) * m_Zp**2
            )
            + gf_R2 * (
                -8*(gDM_L2 - 9*gDM_LR + gDM_R2) * m_f**4 * m_DM**2
                - 4*(2*gDM_L2 + 39*gDM_LR + 2*gDM_R2) * m_f**2 * m_DM**4
                + 4*(gDM_L2 + 18*gDM_LR + gDM_R2) * m_DM**6
                + (2*(gDM_L2 - 3*gDM_LR + gDM_R2)*m_f**4 
                + (-10*gDM_L2 + 3*gDM_LR - 10*gDM_R2)*m_f**2*m_DM**2 
                + (11*gDM_L2 + 6*gDM_LR + 11*gDM_R2)*m_DM**4) * m_Zp**2
            )
        )
        
        b = (
            beta * b_num / 
            (192.0 * np.pi * symm_factor * (m_f**2 - m_DM**2) * (4*m_DM**2 - m_Zp**2)**3)
        )
        
        return a + b * v**2
        # https://arxiv.org/pdf/2005.01515 Eq. (1.26) seems to have an extra factor of 2.

    # ------------------------------------------------------------------
    # Total annihilation rate
    # ------------------------------------------------------------------

    def sigma_v_total(self, kinetic_mixing, v=0.0, symm_factor=1):
        """
        Total annihilation cross section summed over all SM fermions.
        NOTE: It is assumed that electroweak symmetry breaking has
        already occured and SM fermions have their masses.

        Parameters
        ----------
        kinetic_mixing : float
            Kinetic mixing angle epsilon
        v : float
            Relative velocity
        symm_factor : int
            Symmetry factor
        """
        sigma_tot = 0.0

        for f in self.SM_fermions():
            Nc = self.color_factor(f)
            mf = f.mass / 1e3  # Convert MeV to GeV
            
            # Get SM charges for this fermion
            Q_em, T3_L, T3_R = self.get_SM_charges(f)

            # --- Couplings from kinetic mixing ---
            # DM couplings (SM singlet)
            _, (gDM_L, gDM_R) = self.Z_Zprime_couplings(
                kinetic_mixing, Q_em=0.0, T3_L=0.0, T3_R=0.0, QPrime_LR=self.QPrime_DM
            )
            
            # Final state fermion couplings
            _, (gf_L, gf_R) = self.Z_Zprime_couplings(
                kinetic_mixing, Q_em, T3_L, T3_R, QPrime_LR=self.QPrime_SM
            )
            
            sigma_f = self.sigma_v_exact(
                m_f=mf, g_DM=(gDM_L, gDM_R), g_f=(gf_L, gf_R), v=v, symm_factor=symm_factor
            )
            
            sigma_tot += Nc * sigma_f

        return sigma_tot

    

    # ------------------------------------------------------------------
    # Solver for kinetic mixing
    # ------------------------------------------------------------------
    def kinetic_mixing_estimate(self, v, sigmav_target):
        '''
        Estimate for kinetic mixing angle ε based on first-order
        expansion of effective Z' eigenstate couplings.

        Parameters
        ----------

        '''
        sigmav_over_epsilon2 = 0.0

        for f in self.SM_fermions():
            Nc = self.color_factor(f)
            mf = f.mass / 1e3  # Convert MeV to GeV
            
            # SM charges for this fermion
            Q_em, T3_L, T3_R = self.get_SM_charges(f)
            QZ_LR = np.array([T3_L, T3_R]) - sW2 * Q_em

            # --- First-Order Couplings to Z' Eigenstate ---
            g_DM = self.gPrime * self.QPrime_DM
            gf_over_epsilon = -g1*cW**2 * Q_em + g2/cW * sW * QZ_LR

            sigmav_over_epsilon2 += Nc * self.sigma_v_exact(mf, g_DM, gf_over_epsilon, v, symm_factor=1)

        return np.sqrt(sigmav_target / sigmav_over_epsilon2)

    def solve_kinetic_mixing(self, target_sigma_v, v=0.0, symm_factor=1, 
                            epsilon_guess=None, epsilon_min=-0.1, epsilon_max=0.1, 
                            method='auto'):
        """
        Solve for kinetic mixing angle that gives target cross section.
        
        Parameters
        ----------
        target_sigma_v : float
            Target value of <σv> (e.g., 2.5e-10 / 0.12)
        v : float
            Relative velocity
        symm_factor : int
            Symmetry factor
        epsilon_guess : float, optional
            Initial guess for kinetic mixing angle. If provided, Newton-type
            methods are used. If None, analytical estimate is used.
        epsilon_min, epsilon_max : float
            Bounds for kinetic mixing angle search (used for bracketing methods)
        method : str
            Root-finding method. Options:
            - 'auto': Uses 'secant' if epsilon_guess provided, 'brentq' otherwise
            - 'newton': Newton-Raphson (requires epsilon_guess)
            - 'secant': Secant method (requires epsilon_guess)
            - 'brentq', 'brenth', 'ridder': Bracketing methods
            
        Returns
        -------
        result : OptimizeResult
            Solution object containing kinetic mixing value
        """
        def objective(epsilon):
            return self.sigma_v_total(epsilon, v, symm_factor) - target_sigma_v
        
        def kinetic_mixing_estimate(sigma_v):
            """
            Analytical estimate for kinetic mixing from Eq. (1.26) of https://arxiv.org/pdf/2005.01515
            """
            gPrime = self.gPrime
            m_chi = self.m_DM
            m_APrime = self.m_Zprime
            Q_DM = self.QPrime_DM[0]
            Q_SM = 1  # self.QPrime_SM[0]
            
            # Fine structure constants
            alpha = (gPrime * Q_SM)**2 / (4 * np.pi)
            alpha_D = (gPrime * Q_DM)**2 / (4 * np.pi)
            
            # Analytical estimate
            epsilon_est = np.sqrt(
                (4 * m_chi**2 - m_APrime**2)**2 / (16 * np.pi * m_chi**2) 
                * sigma_v / (alpha * alpha_D)
            )
            
            return epsilon_est
        
        # Generate initial guess if not provided
        if epsilon_guess is None:
            # epsilon_guess = kinetic_mixing_estimate(target_sigma_v)
            epsilon_guess = self.kinetic_mixing_estimate(v, target_sigma_v)
            self.kinetic_mixing_guess = epsilon_guess # troubleshooting
        
        # Auto-select method based on whether initial guess is provided
        if method == 'auto':
            method = 'secant' if epsilon_guess is not None else 'brentq'
        
        # Use derivative-based methods if initial guess is provided
        if method in ['newton', 'secant']:
            if epsilon_guess is None:
                raise ValueError(f"Method '{method}' requires epsilon_guess")
            
            result = root_scalar(
                objective,
                x0=epsilon_guess,
                method=method
            )
        else:
            # Use bracketing methods
            result = root_scalar(
                objective, 
                bracket=[epsilon_min, epsilon_max],
                method=method
            )
        
        return result