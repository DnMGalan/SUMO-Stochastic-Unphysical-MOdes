import numpy as np
from qutip import QobjEvo, mesolve
from qutip.solver.parallel import parallel_map


class SHSolver:
    """
    Constructs a stochastic version of the pseudomode-model"""

    def __init__(self):
        # TODO: Define class inputs and initialization logic
        pass

    def compute_coefficients_basis(self, t_corr_list: np.array,
                                     C_list: np.array, n_cut: int) -> np.array:
        """
        Compute the Fourier cosine coefficients of the
        classical bath correlation function. Implements
        Eq. (13) of PRX Quantum 4, 030316 (2023)

        Parameters
        ----------
        t_corr_list : np.ndarray
            Time grid over which the bath correlation
            function `C_list` is defined.
        C_list : np.ndarray
            Discretized values of the classical part of
            the bath correlation function C_class(t).
        n_cut : int
            Truncation order of the Fourier expansion
            (number of modes).

        Returns
        -------
        coeffs : np.ndarray
            Array of Fourier coefficients `c_n` of length `n_cut + 1`.
        """

        T = t_corr_list[-1]
        dt = t_corr_list[1] - t_corr_list[0]
        t_corr_arr = np.asarray(t_corr_list)
        n_vec = np.arange(0, n_cut + 1)
        b = np.pi * t_corr_arr / T
        basis_matrix = np.cos(n_vec[:, None] @ b[None, :])
        product = basis_matrix * C_list
        coeffs = dt * np.sum(product[:, :-1], axis=1) / (2 * T)
        return coeffs

    def generate_A(self, t_corr_list: np.array, coeff_list: np.array,
                   n_cut: int) -> np.array:
        """
        Builds the matrix `A` used to generate the stochastic field ξ(t)
        following Eq. (14) of PRX Quantum 4, 030316 (2023):

        Parameters
        ----------
        t_corr_list : np.ndarray
            Time grid defining the time domain of ξ(t).
        coeff_list : np.ndarray
            Coefficients c_n obtained from `compute_coefficients_basis_2`.
        n_cut : int

        Returns
        -------
        A : np.ndarray
            Real-valued matrix of shape (len(t_corr_list), 2 * n_cut + 1)
            encoding all cosine and sine time functions scaled by √(2 c_n).
        """

        T = t_corr_list[-1]
        n_vec = np.arange(1, n_cut + 1)
        first_col = np.sqrt(coeff_list[0]) * np.ones((len(t_corr_list), 1))
        theta = np.outer(t_corr_list, n_vec) * np.pi / T
        sqrt2 = np.sqrt(2)
        sqrt_coeffs = np.sqrt(coeff_list[1:n_cut + 1])
        cos_terms = sqrt2 * sqrt_coeffs * np.cos(theta)
        sin_terms = sqrt2 * sqrt_coeffs * np.sin(theta)
        interleaved = np.empty((len(t_corr_list), 2 * n_cut))
        interleaved[:, 0::2] = cos_terms
        interleaved[:, 1::2] = sin_terms
        A = np.concatenate([first_col, interleaved], axis=1)

        return A

    def generate_xi_list(self, A: np.ndarray, n_cut: int, n_noise: int,
                         mu: float = 0, sigma: float = 1) -> np.array:
        """
        Generate stochastic realizations of the classical field ξ(t).

        Implements Eq. (14) of PRX Quantum 4, 030316 (2023) by sampling
        independent Gaussian random variables ξ_n with mean `mu` and
        variance `sigma`, then combining them with the deterministic
        basis matrix `A` to form stochastic time-dependent fields ξ(t).

        Parameters
        ----------
        A : np.ndarray
            Basis matrix from `generate_A`.
        n_cut : int

        n_noise : int
            Number of stochastic realizations (samples) to generate.
        mu : float, optional
            Mean of the Gaussian random variables (default is 0).
        sigma : float, optional
            Variance of the Gaussian random variables (default is 1).

        Returns
        -------
        xi_fields : np.ndarray
            Array of shape (n_noise, len(t_corr_list)) containing independent
            realizations of ξ(t).
        """

        xi_lists = np.random.normal(mu, sigma, size=(n_noise, 2 * n_cut + 1))
        xi_fields = xi_lists @ A.T

        return xi_fields

    def generate_fields(self, t_corr_list: np.ndarray, coeff_list: np.ndarray,
                        n_cut: int, n_noise: int) -> np.ndarray:
        """
        Generate a set of ξ(t) trajectories.

        Parameters
        ----------
        t_corr_list : np.ndarray
            Time grid.
        coeff_list : np.ndarray
            c_n coefficients.
        n_cut : int

        n_noise : int
            Number of stochastic trajectories.

        Returns
        -------
        xi_fields : np.ndarray
            Shape (n_noise, M): the generated ξ_k(t).
        """

        A = self.generate_A(t_corr_list, coeff_list, n_cut)
        xi_list = self.generate_xi_list(A, n_cut, n_noise)
        return xi_list

    def one_run(self, k, L, H_xi, xi_list, c_list, t_list, psi0, obs_list,
                args, options):

        """
        Solves the stochastic Lindblad equation Eq. (16) in PRX Quantum 4,
        030316 (2023) for a single realization of ξ(t),
        corresponding to the k-th sample.

        Parameters
        ----------
        k : int
            Index of the ξ_k(t) realization.
        L : qutip.Qobj or qutip.QobjEvo
            System Liouvillian.
        H_xi : qutip.Qobj
            TODO: add a better description
        xi_list : np.ndarray
            All ξ(t) trajectories.
        c_list : list
            Collapse operators for dissipation.
        t_list : np.ndarray
            Integration times.
        psi0 : qutip.Qobj
            Initial state.
        obs_list : list
            Observables to record (passed to `mesolve`).
        args : dict
            Extra arguments for `mesolve`.
        options : qutip.solver.Options
            Solver options.

        Returns
        -------
        result : np.ndarray
            TODO: add better description
        """

        L_m = QobjEvo(
            [L, [H_xi, xi_list[k]]],
            tlist=np.linspace(t_list[0], t_list[-1], len(xi_list[k])),
        )
        result = mesolve(
            L_m, psi0, t_list, c_list, obs_list, args=args, options=options
        ).expect[0]
        return result

    def average_dynamics_parallel(self, L, H_xi, xi_list, c_list, t_list, psi0,
                                  n_noise, obs_list, args, options):
        """
        Runs multiple realizations of the stochastic master equation (Eq. 16)
        in parallel using QuTiP’s `parallel_map`.

        Parameters
        ----------
        L : Qobj
            Lindblad operator or superoperator.
        H_xi : Qobj
            TODO: add better description
        xi_list : np.ndarray
            All ξ(t) trajectories.
        c_list : list
            Collapse operators.
        t_list : np.ndarray
            Time grid for integration.
        psi0 : Qobj
            Initial state.
        n_noise : int
            Number of stochastic realizations.
        obs_list : list
            TODO: add better description
        args : dict
            Arguments passed to QuTiP solvers.
        options : Options
            Solver options.

        Returns
        -------
        dynamics_average : float
            Mean value of the results.
        sigma : float
            Standard deviation of the results.
        dynamics_list : np.ndarray
            Individual trajectory results.
        """

        dynamics_list = parallel_map(
            self.one_run,
            range(n_noise),
            task_args=(L, H_xi, xi_list, c_list, t_list, psi0,
                       obs_list, args, options),
            progress_bar=True)

        dynamics_arr = np.array(dynamics_list)
        dynamics_average = np.mean(dynamics_arr)
        sigma = np.std(dynamics_arr)

        return dynamics_average, sigma, dynamics_list
