import  numpy as np
import cv2
import matplotlib.pyplot as plt

image = np.zeros((700,700,3),dtype=np.uint8)

block_size = 100

for i in range(0,700,block_size):
    for j in range(0,700,block_size):
        image[i,:,:] = (255,255,255)
        image[:,j,:] = (255,255,255)

        # if i != 0 and j != 0 and i != 600 and j != 600 and (i == j or i + j == 600):
        #     image[i:i + block_size,j:j + block_size,:] = (0,0,255)

        top_left = (j,i)
        bottom_right = (j + block_size - 1,i + block_size - 1)
        if i != 0 and i != 600 and (i == j or i + j == 600):
            cv2.rectangle(image,top_left,bottom_right,(0,0,255),-1)
        else:
            cv2.rectangle(image,top_left,bottom_right,(255,255,255),2)


image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.subplot(232)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis("off")

b = image[:,:,0]
g = image[:,:,1]
r = image[:,:,2]

blue_channel = np.zeros((700,700,3),dtype=np.uint8)
green_channel = np.zeros((700,700,3),dtype=np.uint8)
red_channel = np.zeros((700,700,3),dtype=np.uint8)

blue_channel[:,:,0] = b
green_channel[:,:,1] = g
red_channel[:,:,2] = r

blue_channel_rgb = cv2.cvtColor(blue_channel,cv2.COLOR_BGR2RGB)
green_channel_rgb = cv2.cvtColor(green_channel,cv2.COLOR_BGR2RGB)
red_channel_rgb = cv2.cvtColor(red_channel,cv2.COLOR_BGR2RGB)

plt.subplot(234)
plt.imshow(blue_channel_rgb)
plt.title('Blue Channel')
plt.axis("off")

plt.subplot(235)
plt.imshow(green_channel_rgb)
plt.title('Green Channel')
plt.axis('off')

plt.subplot(236)
plt.imshow(red_channel_rgb)
plt.title('Red Channel')
plt.axis('off')

plt.tight_layout()

plt.show()