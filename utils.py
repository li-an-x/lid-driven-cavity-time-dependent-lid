import numpy as np


__all__ = [
    "u_momentum",
    "v_momentum",
    "get_rhs",
    "get_coeff_mat",
    "pressure_correct",
    "update_velocity",
    "check_divergence_free"
]

def u_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, velocity, alpha):
    u_star = np.zeros((imax + 1, jmax))
    d_u = np.zeros((imax + 1, jmax))

    De = mu * dy / dx  # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy

    def A(F, D):
        return max(0, (1 - 0.1 * abs(F / D))**5)

    # compute u_star
    for i in range(1, imax):
        for j in range(1, jmax - 1):
            Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
            Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
            Fn = 0.5 * rho * dx * (v[i, j + 1] + v[i - 1, j + 1])
            Fs = 0.5 * rho * dx * (v[i, j] + v[i - 1, j])

            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i - 1, j] - p[i, j]) * dy

            u_star[i, j] = alpha / aP * ((aE * u[i + 1, j] + aW * u[i - 1, j] + aN * u[i, j + 1] + aS * u[i, j - 1]) + pressure_term) + (1 - alpha) * u[i, j]

            d_u[i, j] = alpha * dy / aP  # refer to Versteeg CFD book

    # set d_u for top and bottom BCs
    for i in range(1, imax):
        j = 0  # bottom
        Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
        Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
        Fn = 0.5 * rho * dx * (v[i, j + 1] + v[i - 1, j + 1])
        Fs = 0

        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = 0
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_u[i, j] = alpha * dy / aP

        j = jmax - 1  # top
        Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
        Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
        Fn = 0
        Fs = 0.5 * rho * dx * (v[i, j] + v[i - 1, j])

        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = 0
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_u[i, j] = alpha * dy / aP

    # Apply BCs
    u_star[0, :jmax] = -u_star[1, :jmax]  # left wall
    u_star[imax, :jmax] = -u_star[imax - 1, :jmax]  # right wall
    u_star[:, 0] = 0.0  # bottom wall
    u_star[:, jmax - 1] = velocity  # top wall

    return u_star, d_u

def v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, alpha):

    v_star = np.zeros((imax, jmax+1))
    d_v = np.zeros((imax, jmax+1))

    De = mu * dy / dx  # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy

    A = lambda F, D: max(0, (1-0.1 * abs(F/D))**5)

    # compute u_star
    for i in range(1, imax-1):
        for j in range(1, jmax):
            Fe = 0.5 * rho * dy * (u[i+1, j] + u[i+1, j-1])
            Fw = 0.5 * rho * dy * (u[i, j] + u[i, j-1])
            Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
            Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i, j-1] - p[i, j]) * dx

            v_star[i, j] = alpha / aP * (aE * v[i+1, j] + aW * v[i-1, j] + aN * v[i, j+1] + aS * v[i, j-1] + pressure_term) + (1-alpha) * v[i, j]

            d_v[i, j] = alpha * dx / aP  # refer to Versteeg CFD book

    # set d_v for left and right BCs
    # Apply BCs
    for j in range(1, jmax):
        i = 0  # left BC
        Fe = 0.5 * rho * dy * (u[i+1, j] + u[i+1, j-1])
        Fw = 0
        Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
        Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = 0
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_v[i, j] = alpha * dx / aP

        i = imax - 1  # right BC
        Fe = 0
        Fw = 0.5 * rho * dy * (u[i, j] + u[i, j-1])
        Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
        Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

        aE = 0
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_v[i, j] = alpha * dx / aP

    # apply BCs
    v_star[0, :] = 0.0  # left wall
    v_star[imax-1, :] = 0.0  # right wall
    v_star[:, 0] = -v_star[:, 1]  # bottom wall
    v_star[:, jmax] = -v_star[:, jmax-1]  # top wall

    return v_star, d_v


def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):

    stride = jmax
    bp = np.zeros((jmax) * (imax))

    # RHS is the same for all nodes except the first one
    # because the first element is set to be zero, it has no pressure correction
    for j in range(jmax):
        for i in range(imax):
            position = i + j * stride
            bp[position] = rho * (u_star[i,j] * dy - u_star[i+1,j] * dy + v_star[i,j] * dx - v_star[i,j+1] * dx)

    # modify for the first element
    bp[0] = 0

    return bp

def get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v):

    N = imax * jmax
    stride = jmax
    Ap = np.zeros((N, N))

    for j in range(jmax):
        for i in range(imax):
            position = i + j * stride
            aE, aW, aN, aS = 0, 0, 0, 0

            # Set BCs for four corners
            if i == 0 and j == 0:
                Ap[position, position] = 1
                continue

            if i == imax-1 and j == 0:
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if i == 0 and j == jmax-1:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if i == imax-1 and j == jmax-1:
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            # Set four boundaries
            if i == 0:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if j == 0:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if i == imax-1:
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if j == jmax-1:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position] = aE + aN + aW + aS
                continue

            # Interior nodes
            Ap[position, position-1] = -rho * d_u[i,j] * dy
            aW = -Ap[position, position-1]

            Ap[position, position+1] = -rho * d_u[i+1,j] * dy
            aE = -Ap[position, position+1]

            Ap[position, position-stride] = -rho * d_v[i,j] * dx
            aS = -Ap[position, position-stride]

            Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
            aN = -Ap[position, position+stride]

            Ap[position, position] = aE + aN + aW + aS

    return Ap

def pressure_correct_vqe(imax, jmax, rhsp, Ap, p, alpha, num_qubits, fid):
    pressure = np.copy(p)  # Initial pressure
    p_prime = np.zeros((imax, jmax))  # Pressure correction matrix
    p_prime_interior = np.linalg.solve(Ap, rhsp) 
    
    sampled_b = rhsp.reshape([2**num_qubits, 1])
    sampled_b = sampled_b /np.linalg.norm(sampled_b)
    M = np.linalg.inv(np.diag(np.diag(Ap)))
    L_new = M@Ap
    b_new = M@sampled_b
    b_new = b_new/np.linalg.norm(b_new)
    Hamiltonian = L_new.transpose()@(np.eye(2**num_qubits)- b_new@b_new.T)@L_new
#     H_op=Operator(Hamiltonian)
    H_op = MatrixOp(H_op).to_pauli_op()
#     print(H_op.num_qubits)

    #normalize classical solution for fidelity calculation:
    classical_solution=p_prime_interior/(np.linalg.norm(p_prime_interior))
    backend=Aer.get_backend('statevector_simulator')
    qi =QuantumInstance(backend, shots=None, seed_simulator=None, max_credits=None, basis_gates=None,                     coupling_map=None, initial_layout=None, 
                    pass_manager=None, bound_pass_manager=None, seed_transpiler=None, 
                    optimization_level=None, backend_options=None, noise_model=None, 
                    timeout=None, wait=5.0, skip_qobj_validation=True,                                                   measurement_error_mitigation_cls=None, 
                    cals_matrix_refresh_period=30, measurement_error_mitigation_shots=None, 
                    job_callback=None, mit_pattern=None, max_job_retries=50)
    ansatz = RealAmplitudes(num_qubits=4, entanglement='linear', reps=N+1, 
                            skip_unentangled_qubits=False, skip_final_rotation_layer=False, 
                            parameter_prefix='θ', insert_barriers=False, initial_state=None, 
                            name='RealAmplitudes')
    rng = np.random.default_rng(581)
    initial_point = rng.uniform(0, 2*np.pi, ansatz.num_parameters)
    optimizer  = L_BFGS_B()
    vqe = VQE(ansatz, optimizer,initial_point = initial_point, callback=store_intermediate_result,                quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(H_op)
    quantum_solution_normal = result.eigenstate.real/(np.linalg.norm(result.eigenstate.real))
    fidcheck = (quantum_solution_normal.T@classical_solution)**2
    fid.append(fidcheck)
    p_prime_interior = result.eigenstate.real

    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime
def pressure_correct(imax, jmax, rhsp, Ap, p, alpha):
    pressure = np.copy(p)  # Initial pressure
    p_prime = np.zeros((imax, jmax))  # Pressure correction matrix
    p_prime_interior = np.linalg.solve(Ap, rhsp) 
#     p_prime_interior = p_prime_interior + np.random.normal(1e-2, 1, len(p_prime_interior))


    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime

def pressure_correct1(imax, jmax, rhsp, Ap, p, alpha):
    pressure = np.copy(p)  # Initial pressure
    p_prime = np.zeros((imax, jmax))  # Pressure correction matrix
    p_prime_interior = np.linalg.solve(Ap, rhsp) 
    p_prime_interior = p_prime_interior + np.random.normal(0, 1e-2, len(p_prime_interior))


    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime

def pressure_correct2(imax, jmax, rhsp, Ap, p, alpha):
    pressure = np.copy(p)  # Initial pressure
    p_prime = np.zeros((imax, jmax))  # Pressure correction matrix
    p_prime_interior = np.linalg.solve(Ap, rhsp) 
    p_prime_interior = p_prime_interior + np.random.normal(0, 1e-5, len(p_prime_interior))


    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime
def pressure_correct3(imax, jmax, rhsp, Ap, p, alpha):
    pressure = np.copy(p)  # Initial pressure
    p_prime = np.zeros((imax, jmax))  # Pressure correction matrix
    p_prime_interior = np.linalg.solve(Ap, rhsp) 
    p_prime_interior = p_prime_interior + np.random.normal(0, 1e-3, len(p_prime_interior))


    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime
def pressure_correct4(imax, jmax, rhsp, Ap, p, alpha):
    pressure = np.copy(p)  # Initial pressure
    p_prime = np.zeros((imax, jmax))  # Pressure correction matrix
    p_prime_interior = np.linalg.solve(Ap, rhsp) 
    p_prime_interior = p_prime_interior + np.random.normal(0, 1e-4, len(p_prime_interior))


    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime

def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity,iteration, y):
    u = np.zeros((imax+1, jmax))
    v = np.zeros((imax, jmax+1))

    # Update interior nodes of u and v
    for i in range(1, imax):
        for j in range(1, jmax-1):
            u[i,j] = u_star[i,j] + d_u[i,j] * (p_prime[i-1,j] - p_prime[i,j])

    for i in range(1, imax-1):
        for j in range(1, jmax):
            v[i,j] = v_star[i,j] + d_v[i,j] * (p_prime[i,j-1] - p_prime[i,j])

    # Update BCs
    v[0,:] = 0.0          # left wall
    v[imax-1,:] = 0.0     # right wall
    v[:,0] = -v[:,1]      # bottom wall
    v[:,-1] = -v[:,-2]    # top wall

    u[0,:] = -u[1,:]      # left wall
    u[imax,:] = -u[imax-1,:] # right wall
    u[:,0] = 0.0          # bottom wall
    u[:,-1] = y[iteration]    # top wall

    return u, v


def check_divergence_free(imax, jmax, dx, dy, u, v):
    div = np.zeros((imax, jmax))

    for i in range(imax):
        for j in range(jmax):
            div[i, j] = (1/dx) * (u[i, j] - u[i+1, j]) + (1/dy) * (v[i, j] - v[i, j+1])

    return div





