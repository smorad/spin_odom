import cv2
import math
import numpy as np
import scipy.stats
from scipy.misc import factorial
import glob
from matplotlib import pyplot as plt
from sklearn import linear_model


class AngleEstimator:

    # Estimates rotation axis given images

    def __init__(self, img_dir, img_start=0, img_count=9001):
        self.img_paths = sorted(glob.glob(img_dir+'/*.jpg'))[img_start:img_count]
        print('Loaded', len(self.img_paths), 'frames')
        print('\n'.join(self.img_paths))
        #print(self.img_paths)


    def likelihood(theta, n, x):
        return (factorial(n) / (factorial(x) * factorial(n - x))) \
                * (theta ** x) * ((1 - theta) ** (n - x))

    def estimate(self, debug=False):
        # bayesian updating
        mu, sigma = self.find_angle(cv2.imread(self.img_paths[0],0))
        all_mus = []
    
        for i, path in enumerate(self.img_paths[1:]):
            #print('Frame', i, mu, sigma)
            m, s = self.find_angle(cv2.imread(path,0))
            if m == None:
                continue
            all_mus += [m]
            if s < 0 or s > 1:
                continue
            #print('sigma', s)
            mu = (sigma*s + math.e**2 * mu) / (sigma + math.e**2)
            sigma = (sigma*math.e**2) / (sigma + math.e**2)

        #mu = math.pi/2 - mu
        #print('final data', mu, sigma)
        med_mus = np.median(all_mus)
        #median_abs_dev = np.median(np.abs(np.array(all_mus) - med_mus))
        med_sigma = np.std(all_mus)
        print('med final data', np.median(all_mus))
        
        if debug:
            img = cv2.imread(self.img_paths[-1],0)
            ((x1,y1), (x2,y2))= self.to_line(480/2, med_mus)
            cv2.line(img, (x1,y1), (x2,y2), 0, 2)
            ((x1,y1), (x2,y2))= self.to_line(480/2, med_mus+med_sigma)
            cv2.line(img, (x1,y1), (x2,y2), 32, 2)
            ((x1,y1), (x2,y2))= self.to_line(480/2, med_mus-med_sigma)
            cv2.line(img, (x1,y1), (x2,y2), 32, 2)
#            img = cv2.imread(self.img_paths[-1],0)
#            ((x1,y1), (x2,y2))= self.to_line(480/2, mu)
#            cv2.line(img, (x1,y1), (x2,y2), 0, 2)
#            ((x1,y1), (x2,y2))= self.to_line(480/2, mu+sigma)
#            cv2.line(img, (x1,y1), (x2,y2), 32, 2)
#            ((x1,y1), (x2,y2))= self.to_line(480/2, mu-sigma)
#            cv2.line(img, (x1,y1), (x2,y2), 32, 2)
            plt.figure()
            plt.imshow(img)
            plt.title('Estimate and Variance Overlaid on Frame')
            #plt.show()
            plt.savefig('results/psf_overlay.eps', format='eps')
            plt.figure()
            plt.hist(all_mus, 20)
            plt.title('Per-Frame $\\theta$ Estimates')
            plt.savefig('results/psf_hist.eps', format='eps')
            #plt.show()
        
        return np.median(all_mus) - math.pi/2, med_sigma

    def to_line(self, rho, theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 - 1000*b)
        y1 = int(y0 + 1000*a)
        x2 = int(x0 + 1000*b)
        y2 = int(y0 - 1000*a)
        return ((x1,y1), (x2,y2))


    def to_lines(self, h):
        lines = []
        mean_angle = np.mean(h[:,0][:,1])
        mean_rho = np.mean(h[:,0][:,0])
        for rho, theta in h[:,0]:
            lines += [self.to_line(rho, theta)]

#        a = np.cos(mean_angle)
#        b = np.sin(mean_angle)
#        x0 = a*mean_rho
#        y0 = b*mean_rho
#        x1 = int(x0 - 1000*b)
#        y1 = int(y0 + 1000*a)
#        x2 = int(x0 + 1000*b)
#        y2 = int(y0 - 1000*a)
#        lines += [(x1,y1), (x2,y2)]

        return lines



    def debug_angles(self, img, h):


            cv2.line(img, (x1,y1), (x2,y2), 0, 2)
        
        # for mean

        

    def find_angle(self, img, debug=False):
        edges = cv2.Canny(img,100,100)
        #h = cv2.HoughLines(edges,2,0.05,100)
        h = cv2.HoughLines(edges,2,math.pi/180,80)
        if h is None:
            return None, None
        angles = h[:,0][:,1]
        rhos = h[:,0][:,0]

        if debug:
            self.debug_angles(img, h)


        #median_angle = np.median(angles)
        #median_abs_dev = 1.4826 * np.median(np.abs(angles - np.median(angles)))

        #mean_angle = np.mean(angles)
        #mean_rho = np.mean(rhos)
        #plt.imshow(edges)

        # should be -? worked with - before on one frame
        mean,stddev = scipy.stats.norm.fit(angles)
        #mean += math.pi/2 # hough measure theta=0 as vertical line
        #print('got mean', mean, 'dev', stddev)
        return mean, stddev

    def find_angle2(self, img, debug=False):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        mag_spectrum = 20*np.log(np.abs(fshift)).astype('uint8')

        #plt.imshow(mag_spectrum)
        #plt.show()
        h = cv2.HoughLines(mag_spectrum,10, 1, 200)

        
        
        for rho, theta in h[:,0]:
            # throw out bad ones
            #if rho - math.pi/2 < 0.1 or rho - math.pi < 0.1 or rho < 0.1:
            if abs(theta % (math.pi/2)) < 0.02 or abs(theta) < 0.02:
                #print('skip')
                continue
            ((x1,y1), (x2,y2)) = self.to_line(rho, theta)
            #cv2.line(mag_spectrum, (x1,y1), (x2,y2), 255, 1)
            cv2.line(img, (x1,y1), (x2,y2), 255, 2)
            #print(rho, theta)
        #lines = self.to_lines(h)
        #print(len(lines))
        #for ((x1,y1), (x2,y2)) in lines:
        #    cv2.line(mag_spectrum, (x1,y1), (x2,y2), 0, 20)

        #print(img)
        #plt.imshow(mag_spectrum)
        #plt.show()
        #plt.imshow(img)
        #plt.show()
        print(h)
