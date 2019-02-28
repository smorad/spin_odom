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


class AngleEstimator:

    # Estimates rotation axis given images

    def __init__(self, img_dir, img_start=0, img_count=9001):
        self.img_paths = sorted(glob.glob(img_dir+'/*.jpg'))[img_start:img_count]
        print('Loaded', len(self.img_paths), 'frames')
        #print('\n'.join(self.img_paths))
        #print(self.img_paths)

    def stitch(self):
        right_offset = 10
        frames = len(self.img_paths[-right_offset:]) - 1
        xrate, yrate = self.px_rate_per_frame
        max_rows = int(480 + yrate*frames)
        max_cols = int(640 + xrate*frames)
        pano = np.ndarray((max_rows, max_cols), dtype=float)
        #psf = self.psf2(np.linalg.norm(self.px_rate_per_frame))
        for i, path in enumerate(self.img_paths[-right_offset:]):
            trans = [int(xrate * i), int(yrate * i)]
            img = cv2.imread(path, 0)
            #img = restoration.richardson_lucy(img, psf)
            #img = restoration.wiener(img, psf, 1/5000)
            #print(trans)
            pano[trans[1]:trans[1] + 480, trans[0]:trans[0] + 640] = img
        #print(pano)
        plt.imshow(pano, cmap='gray')
        plt.show()



    def deblur_frame(self, idx):
        img = cv2.imread(self.img_paths[idx], 0)
        img = np.divide(img, np.max(img))
        psf = self.psf2(np.linalg.norm(self.px_rate_per_frame))
        #img2 = restoration.richardson_lucy(img, psf)
        img2, _ = restoration.unsupervised_wiener(img, psf)

        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img2)
        plt.show()

    def psf2(self, dist):
        d = int(dist)
        #sz = 10#int(self.px_rate_per_frame[0])
        szx = int(self.px_rate_per_frame[0] / 50)#10
        szy = int(self.px_rate_per_frame[1])#2
        angle = self.angle
        kern = np.ones((1, d), np.float32)
        c, s = np.cos(angle), np.sin(angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        szx2 = szx // 2
        szy2 = szy // 2
        A[:,2] = (szx2, szy2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
        kern = cv2.warpAffine(kern, A, (szx, szy), flags=cv2.INTER_CUBIC)
        kern = kern / np.sum(kern)
        plt.figure()
        plt.imshow(kern)
        plt.show()
        # correct, dont transpose
        return kern

    def psf(self, dist, diam=10):
        # TODO fix
        diam = int(dist)
        dist = int(dist)
        kernel = np.ones((1, dist), np.float32)
        c, s = np.cos(self.angle), np.sin(self.angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        diam2 = int(diam / 2)
        A[:,2] = (diam2, diam2) - np.dot(A[:,:2], ((dist-1)*0.5, 0))
        kernel = cv2.warpAffine(kernel, A, (diam, diam), flags=cv2.INTER_CUBIC)
        print('k orig', kernel)
        #kernel = kernel * 255
        kernel = np.divide(kernel, np.sum(kernel))
        print('sum', np.sum(kernel))
        #print('kernel', kernel)
        print('kmax should be less than 127:', np.max(kernel))
        #kernel = np.int8(kernel) # to 0,255 fmt
        #print('kernel', kernel)
        plt.figure()
        plt.imshow(kernel)
        plt.show()
        return kernel.transpose()

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

                if slope - math.tan(self.angle) < 0.000001: # TODO should be bigger
                    good_lengths.append(math.sqrt((y1 - y0)**2 + (x1-x0)**2))
                    #print(good_lengths[-1])

        print('Found', len(good_lengths), 'good SIFT samples')
        dist = np.median(good_lengths)


        #final_rate = dist_rad * fps
        final_rate = self.px_to_rad(dist) * fps
        self.rate = final_rate

        print('final rate', final_rate)
        return final_rate


    def px_to_rad(self, dist_px):
        h_fov = math.radians(15.55) * 2 # measured from center out
        v_fov = math.radians(12.20) * 2

        px_dist_x = dist_px * math.cos(self.angle)
        px_dist_y = dist_px * math.sin(self.angle)

        rad_per_px_x = (h_fov / 640)
        rad_per_px_y = (v_fov / 480)

        self.px_rate_per_frame = [px_dist_x, px_dist_y]

        #dist_rad = rad_per_px_x * px_dist_x + rad_per_px_y * px_dist_y 
        dist_rad = math.sqrt((px_dist_x * rad_per_px_x)**2 + (px_dist_y * rad_per_px_y)**2)  
        return dist_rad
            
        
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
