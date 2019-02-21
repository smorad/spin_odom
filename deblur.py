import cv2
import math
import numpy as np
import scipy.stats
from scipy.misc import factorial
import glob
from matplotlib import pyplot as plt
from sklearn import linear_model
from scipy.signal import convolve2d
from skimage import restoration
from skimage import feature


class Deblur:
    # see https://github.com/opencv/opencv/blob/master/samples/python/deconvolution.py
    def __init__(self, angle_estimator, width=640, height=480):
        self.est = angle_estimator
        self.angle = self.est.estimate()[0]
        self.w = width
        self.h = height

    def deblur(self):
        img = cv2.imread(self.est.img_paths[0],0)
        img = img / 255
        new_img = self.restore(img, 80)
        plt.figure()
        plt.imshow(img)
        plt.title('before')
        plt.figure()
        plt.imshow(new_img)
        plt.title('after')
        plt.show()

    def restore(self, img, dist):
        psf = self.psf(dist)
        #restored = restoration.wiener(img, psf, 1100)
        restored, _ = restoration.unsupervised_wiener(img, psf)

        return restored


    def psf(self, dist, diam=80):
        dims = (dist, dist * self.angle) # dimension of filter
        print(dims)
        kernel = np.ones((1, dist), np.float32)
        c, s = np.cos(self.angle), np.sin(self.angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        diam2 = int(diam / 2)
        A[:,2] = (diam2, diam2) - np.dot(A[:,:2], ((dist-1)*0.5, 0))
        kernel = cv2.warpAffine(kernel, A, (diam, diam), flags=cv2.INTER_CUBIC)
        kernel = kernel * 127
        print('kernel', kernel)
        print('kmax should be less than 127:', np.max(kernel))
        #kernel = np.int8(kernel) # to 0,255 fmt
        print('kernel', kernel)
        plt.figure()
        plt.imshow(kernel)
        return kernel
                


class AngleEstimator:

    # Estimates rotation axis given images

    def __init__(self, img_dir, img_start=0, img_count=9001):
        self.img_paths = sorted(glob.glob(img_dir+'/*.jpg'))[img_start:img_count]
        print('Loaded', len(self.img_paths), 'frames')
        #print('\n'.join(self.img_paths))
        #print(self.img_paths)



    def estimate(self, debug=False):
        # bayesian updating
        all_mus = []
    
        for i, path in enumerate(self.img_paths[1:]):
            #print('Frame', i, mu, sigma)
            m, s = self.find_angle(cv2.imread(path,0))
            if m == None:
                continue
            all_mus += [m]
            if s < 0 or s > 1:
                continue
        med_mus = np.median(all_mus)
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
            plt.figure()
            plt.imshow(img)
            plt.title('Estimate and Variance Overlaid on Frame')
            #plt.show()
            #plt.savefig('results/psf_overlay.eps', format='eps')
            plt.figure()
            plt.hist(all_mus, 20)
            plt.title('Per-Frame $\\theta$ Estimates')
            #plt.savefig('results/psf_hist.eps', format='eps')
            #plt.show()
        
        self.angle = np.median(all_mus) - math.pi/2
        return self.angle, med_sigma

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


        return lines



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


    def estimate_rate_march(self):
        rates = []
        for path in self.img_paths:
            img = cv2.imread(path, 0)
            rate = self.march_psf_length(img)
            if rate:
                rates += [rate]

        med_rate = np.median(rates)
        #line_readout_time = 11.081 / 480
        #px_readout_time = line_readout_time / 640
        line_exposure_time = 0.0011081
        fov = math.radians(15.55)
        theta_step = fov / 640 # rad per px 
        final_rate = theta_step * line_exposure_time * med_rate
        print('final rate', final_rate)


    def march_psf_length(self, img):
        # find edges
        # march along smear direction
        # max is first pixel that is less than 90% of the prev pixel
        # min is the first pixel greater than 90% of the prev pixel
    
        # do hough
        # count along hough lines

        # WONT WORK
        # but can give us scale with imu

        lengths = []
        edges = cv2.Canny(img, 100, 100)
        if edges is None:
            return None
        h = cv2.HoughLinesP(edges,2,math.pi/180,100, minLineLength=30, maxLineGap=5)
        if h is None:
            return None

        for [res] in h:
            x0, y0, x1, y1 = res
            length = math.sqrt((y1 - y0)**2 + (x1 - x0)**2)
            lengths += [length]
        print(np.mean(lengths))
        return np.mean(lengths)
            #m = (y1 - y0) / (x1 - x0)
            #b = y0
            #xs = np.linspace
            #inverse = (1 / m) * x + b



#            xs = range(0, 640)
#            ys = [int(math.tan(angles[i] - math.pi/2) * x + rhos[i]) for x in xs] # y=mx+b
#            indices = [xs, ys]
#            # remove off screen values
#            indices = [idx for idx in indices if idx[1] < 480 and idx[1] > 0]
#            
#            plt.figure()
#            plt.imshow(edges)
#            plt.figure()
#            plt.plot(xs, ys)
#            plt.xlim(0, 640)
#            plt.ylim(480, 0)
#            plt.show()
#            #for j in range(len(xs - 1)):
#            comparisons = zip(indices, indices[1:])
#            for a, b in comparisons:
#                # edge found, start counting
#                if edges[a[1]][a[0]]:
#                    print('edges')
#                    pixels = 0
#                    # check one above and below
#                    if img[b[1]][b[0]] < 0.9 * img[a[1]][a[0]] \
#                            or img[b[1]]: # if prev image < cur image 
#                        pixels += 1
#                    else:
#                        lengths += [pixels]
#                        pixels = 0
#        print(lengths) 
#        
#


        #def fn(x, b):
        #    return int(math.tan(self.angle) * x + b)

#        psf_lengths = []
#        edges = cv2.Canny(img,100,100)
#        for b in range(480):
#            # y = mx + b
#            counting = False
#            for x in range(640-1):
#                y = fn(x, b)
#                if y >= 475: # TODO fix this shit
#                    continue
#                pixels = 0
#                if edges[y][x]:
#                    # TODO Handle edge conditions
#                    counting = True
#                if counting:
#                    pixels += 1
#                    old_value = img[y][x]
#                    new_value = img[fn(x+1,b)][x+1]
#
#                    if new_value < 0.2 * old_value:
#                        psf_lengths += [pixels]
#                        pixels = 0
#                        counting = False
#
#        print('lengths', psf_lengths)
#        print('frame med', np.median(psf_lengths))
#        return np.median(psf_lengths)


    def estimate_rate_sift(self):
        window_frames = 50
        fps = 90
        results = []
        #orb = cv2.ORB()
        orb = cv2.ORB_create()
        points = []
        descriptors = []
        for i in range(window_frames):
            kp, ds = orb.detectAndCompute(cv2.imread(self.img_paths[i],0), None)
            points += [kp]
            descriptors += [ds]

        #print(descriptors)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = []
        for i in range(len(descriptors)):
            for j in range(i+1, len(descriptors)):
                if descriptors[i] is None  or descriptors[j] is None:
                    continue
                match = (i, j, sorted(matcher.match(descriptors[i], descriptors[j]), key=lambda x:x.distance)[0].distance)
                matches += [match]

        #best = sorted(matches, key=lambda x: x[2]) 
        median = sorted(matches, key=lambda x: x[1] - x[0])
        print(median)
        print(median[int(len(median) / 2)])


    def estimate_rate_seq_frames(self):
        fps = 90
        h_fov = math.radians(15.5) * 2 # measured from center out
        good_lengths = []
        for f0, f1 in zip(self.img_paths, self.img_paths[1:]):
            orb = cv2.ORB_create()
            img0 = cv2.imread(f0, 0)
            img1 = cv2.imread(f1, 0)
            kp0, ds0 = orb.detectAndCompute(img0, None)
            kp1, ds1 = orb.detectAndCompute(img1, None)
            if ds0 is None or ds1 is None:
                continue
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(ds0, ds1)


            for m in matches:
                idx0 = m.queryIdx
                idx1 = m.trainIdx

                (x0, y0) = kp0[idx0].pt
                (x1, y1) = kp1[idx1].pt

                slope = (y1 - y0) / (x1 - x0)

                if slope - math.tan(self.angle) < 0.1:
                    good_lengths.append(math.sqrt((y1 - y0)**2 + (x1-x0)**2))
                    print(good_lengths[-1])

        dist = np.median(good_lengths)
        rad_per_px = h_fov / 640 

        print('rate', dist * rad_per_px * fps)
            
        

            #out = cv2.drawMatches(img0, kp0, img1, kp1, matches, None)
            #plt.imshow(out)
            #plt.show()

            
    def estimate_rate(self):
        window_frames = 40
        base_frames = 5
        fps = 90
        results = []
        for i in range(base_frames):
            base_img = cv2.imread(self.img_paths[i],0)
            cmp_frames = range(i+1, i+window_frames)
            for j in cmp_frames:
                cmp_img = cv2.imread(self.img_paths[j],0)
                shift, error, diffphase = feature.register_translation(base_img, cmp_img)
                if diffphase < 0:
                    # not interested in inverse matches
                    error = 10000
                results += [(error, shift, i, j)]

        rates = []
        for candidate in results:
            error, shift, base_idx, cmp_idx = candidate
            rate = (cmp_idx - base_idx) * (1/fps) * 2 * math.pi
            rates += [rate]

        print(rates, np.mean(rates))
        plt.figure()
        plt.imshow(cv2.imread(self.img_paths[base_idx],0))
        plt.figure()
        plt.imshow(cv2.imread(self.img_paths[cmp_idx],0))
        plt.show()

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
                continue
            ((x1,y1), (x2,y2)) = self.to_line(rho, theta)
            cv2.line(img, (x1,y1), (x2,y2), 255, 2)
        print(h)
