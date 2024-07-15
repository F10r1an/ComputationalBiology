def solve(f_quad, v_0, t):
    v = np.zeros(np.shape(t))
    v[0] = v_0
    for i, t_i in enumerate(t[:-1]):
        j = i+1
        t_j=t[j]
        v_j = odeint(f_quad,v[i],[t_i,t_j])[1]
        if v_j>v_peak:
            v_j=v_reset
        v[j] = v_j
    return v