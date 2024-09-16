import cv2
from matplotlib import pyplot as plt
import numpy as np

# remove boarder
# remove water column
# mean filter
# edge detection

class pre_process:
    def remove_border(self,img):
        img = img[:, 55:img.shape[1] - 55]
        return img

    def remove_water_column(self,img,plot=False):
        average_intensity_per_row = np.mean(img, axis=1)

        row=0
        cumulative_sum = np.cumsum(average_intensity_per_row)
        last = cumulative_sum[-1]

        for i in range(100,len(cumulative_sum)-10,5):
            avg = 0
            for j in range(5):
                avg+=cumulative_sum[i+j]
            avg = avg/5
            if last - avg < 500:
                row=i
                break

        if plot:
            plt.subplot(1, 2, 1)
            plt.plot(average_intensity_per_row, color='blue')
            plt.title('Average Intensity per Row')
            plt.xlabel('Row Index')
            plt.ylabel('Average Intensity')
            plt.xlim(0, img.shape[0])
            plt.ylim(0, 255)

            # Cumulative Sum of Average Intensity per Row
            plt.subplot(1, 2, 2)
            plt.plot(cumulative_sum, color='red')
            plt.title('Cumulative Sum of Average Intensity per Row')
            plt.xlabel('Row Index')
            plt.ylabel('Cumulative Sum')
            plt.xlim(0, img.shape[0])
            plt.ylim(0, cumulative_sum.max())
            plt.axvline(x=row, color='green', linestyle='--', linewidth=1.0)

        img = img[:row, :]

        return img

    def mean_filter(self,img, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        img = cv2.filter2D(img, -1, kernel)
        return img

    def edge_detection(self,img):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        img = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
        return img
    
    def totalPipeline(self,img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = self.remove_border(img)
        img = self.remove_water_column(img)
        img = self.mean_filter(img, 5)
        img = self.edge_detection(img)
        img = cv2.resize(img, (410, 410))
        return img

if __name__ == "__main__":

    img1 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    pp = pre_process()
    img = pp.remove_border(img1)
    img = pp.remove_water_column(img)
    img = pp.mean_filter(img, 5)
    img = pp.edge_detection(img)
    img = cv2.resize(img, (410, 410))
    print(img.shape)
    # pp.totalPipeline('d1/camera/sonar/1.jpg')


    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Sonar Image1')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img, cmap='gray')
    plt.title('Pre-Proccessed ')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

