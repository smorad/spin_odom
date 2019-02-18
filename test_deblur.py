from deblur import AngleEstimator
import math
from matplotlib import pyplot as plt
#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170536_12218', 625, 1000)
#a.estimate()


#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 900, 905)
#a.estimate(debug=True)
def run():
    for i in range(10):
        offset = 1120
        a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', offset+i, offset+i+1)
        a.estimate(debug=True)
    


def benchmark():
    meds = []
    sigs = []
    error = []
    xs = range(100)
    for i in xs:
        offset = 1120
        a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', offset, offset+i+1)
        if i < len(xs) - 1:
            m, s = a.estimate()
        else:
            a.estimate(debug=True)

        # + instead of - because img coordinate system is left-handed
        error += [abs(m + math.atan2(203 - 240, 568 - 219))]
        meds += [m]
        sigs += [s]

    plt.figure()
    plt.plot(xs, meds)
    plt.title('PSF Direction')
    plt.xlabel('Image Frame')
    plt.ylabel('$\\theta$ (rad)')
    plt.savefig('results/psf_direction_median.eps', format='eps')

    plt.figure()
    plt.plot(xs, sigs)
    plt.title('PSF Standard Deviation')
    plt.xlabel('Image Frame')
    plt.ylabel('$\sigma$')
    plt.savefig('results/psf_direction_stddev.eps', format='eps')

    plt.figure()
    plt.plot(xs, error)
    plt.title('PSF Absolute Error')
    plt.xlabel('Image Frame')
    plt.ylabel('$|\\theta_e - \\theta_a|$')
    plt.savefig('results/psf_direction_error.eps', format='eps')

    
#run()
benchmark()
