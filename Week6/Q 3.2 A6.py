# Assignment 6


# Packages


# Scale-space edge detector


def Exercise_3_2():

    def H(tau, sigma=1):
        return 1 / ((4 * (np.square(np.pi) * (sigma**2 + tau**2)**2)))

    t = np.linspace(-0.5, 5, 100)  # Only positive values of tau.
    sigmas = [1, 1.1, 1.2, 1.3, 1.4, 1.5]
    Zs = [Z(t, sigma=s) for s in sigmas]
    Ps = [0, 0, 0, 0, 0, 0]

    # Plotting.
    plt.plot(t, Zs[0], 'r', label="$\sigma = 1.0$")
    plt.plot(t, Zs[1], 'b', label="$\sigma = 1.1$")
    plt.plot(t, Zs[2], 'g', label="$\sigma = 1.2$")
    plt.plot(t, Zs[3], 'orange', label="$\sigma = 1.3$")
    plt.plot(t, Zs[4], 'lime', label="$\sigma = 1.4$")
    plt.plot(t, Zs[5], 'purple', label="$\sigma = 1.5$")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.xticks(np.arange(0, 5, step=0.4))
    plt.xlabel("$\\tau$")
    plt.ylabel("$Z(\\tau)$")
    plt.tight_layout()
    plt.show()
