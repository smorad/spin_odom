from deblur import AngleEstimator, Deblur
import math
from matplotlib import pyplot as plt
#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170536_12218', 625, 1000)
#a.estimate()


#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 900, 905)
#a.estimate(debug=True)
def run():
    a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 1100, 1114)
    a.estimate()
    est_rate = a.estimate_rate_seq_frames()
    #actual_rate = (37 / 90) * 2*math.pi
    actual_rate = 2*math.pi / (37 / 90) + 0.0771790049385805 # offset from slight shift
    print('actual rate rad/s', actual_rate)
    print('pct diff', (actual_rate - est_rate) / actual_rate * 100)
    a.stitch()
    #d = Deblur(a) 
    #d.deblur()


def benchmark():
    meds = []
    sigs = []
    error = []
    rates = []
    rate_errors = []
    xs = range(100)
    for i in xs:
        offset = 1120
        a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', offset, offset+i+1)
        if i < len(xs) - 1:
            m, s = a.estimate()
        else:
            a.estimate(debug=True)

        # + instead of - because img coordinate system is left-handed
        error += [m + math.atan2(203 - 240, 568 - 219)]
        #error += [abs(m + 0.1024)] # see matlab for how we got number 
        meds += [m]
        sigs += [s]
        
        # rates
        est_rate = a.estimate_rate_seq_frames()
        actual_rate = 2*math.pi / (37 / 90) + 0.0771790049385805 # offset from slight shift
        rates += [est_rate]
        rate_errors += [actual_rate - est_rate]

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
    plt.ylabel('$\sigma$ (rad)')
    plt.savefig('results/psf_direction_stddev.eps', format='eps')

    plt.figure()
    plt.plot(xs, error)
    plt.title('PSF Direction Error')
    plt.xlabel('Image Frame')
    plt.ylabel('$\\theta_a - \\theta_e$ (rad)')
    plt.savefig('results/psf_direction_error.eps', format='eps')

    plt.figure()
    plt.plot(xs, rates)
    plt.title('Spin Rate')
    plt.xlabel('Image Frame')
    plt.ylabel('$\dot{\\theta}$ (rad/s)')
    plt.savefig('results/rate.eps', format='eps')


    plt.figure()
    plt.plot(xs, rate_errors)
    plt.title('Spin Rate Error')
    plt.xlabel('Image Frame')
    plt.ylabel('$\dot{\\theta_a} - \dot{\\theta_e}$ (rad/s)')
    plt.savefig('results/rate_error.eps', format='eps')
    
run()
#benchmark()
