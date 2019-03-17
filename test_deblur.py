from deblur import AngleEstimator 
import math
from matplotlib import pyplot as plt
#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170536_12218', 625, 1000)
#a.estimate()


#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 900, 905)
#a.estimate(debug=True)

def deblur():
    a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 1100, 1114)
    a.estimate()
    a.estimate_rate_seq_frames()
    a.deblur_frame(0)

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


def benchmark(path, actual_slope, actual_rate, frames=100, offset=0):
    meds = []
    sigs = []
    error = []
    rates = []
    rate_errors = []
    xs = range(frames)
    for i in xs:
        #offset = 1100
        #a = AngleEstimator('/home/smorad/spin_odom/images_new/speer_data_new/20190128_172606_19738', offset, offset+i+1)
        a = AngleEstimator(path, offset, offset+i+1)
        if i < len(xs) - 1:
            m, s = a.estimate()
        else:
            a.estimate(debug=True)

        # + instead of - because img coordinate system is left-handed
        #error += [abs(m) - abs(-0.090)]
        error += [m - actual_slope]
        meds += [m]
        sigs += [s]
        
        # rates
        est_rate = a.estimate_rate_seq_frames()
        #actual_rate = 2*math.pi / (37 / 90) + 0.0771790049385805 # offset from slight shift
        rates += [est_rate]
        rate_errors += [actual_rate - est_rate]

    print(rates)

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
    

def vicon(path, begin, frames):
    a = AngleEstimator(path, begin, begin+frames)
    #a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 1100, 1200)
    a.estimate()
    est_rate = a.estimate_rate_seq_frames()
    #actual_rate = (37 / 90) * 2*math.pi
    #actual_rate = 2*math.pi / (37 / 90) + 0.0771790049385805 # offset from slight shift
    #print('actual rate rad/s', actual_rate)
    #print('pct diff', (actual_rate - est_rate) / actual_rate * 100)
    a.stitch()
    #d = Deblur(a) 
    #d.deblur()


#run()
#deblur()
#benchmark()
#vicon()
#benchmark()
        #actual_rate = 20.7608
        #error += [abs(m) - abs(-0.090)]
#benchmark('/home/smorad/spin_odom/images_new/speer_data_new/empty', -0.0089, 20.7608, 45, 1100) # empty
#benchmark('/home/smorad/spin_odom/images_new/speer_data_new/chairs', 0.0154, 8.1810, 20, 800)
#benchmark('/home/smorad/spin_odom/images_new/speer_data_new/pillars', -0.0462, 13.0839, 45, 970)
#benchmark('/home/smorad/spin_odom/images_new/speer_data_new/pillars', -0.0094203, 13.106, 20, 970)



vicon('/home/smorad/spin_odom/images_new/speer_data_new/empty', 1120, 45) # empty
#vicon('/home/smorad/spin_odom/images_new/speer_data_new/chairs', 800, 20)
#vicon('/home/smorad/spin_odom/images_new/speer_data_new/pillars', 955, 20)
