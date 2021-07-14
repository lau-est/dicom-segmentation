from dicom_utils import reader, display, transform
from matplotlib import pyplot as plt

slices = reader.read_ct("data/input/COD0005")#("C:\\Users/farah/Desktop/COVID-19/Healthy/2", select_only_size=1024)
print("Slices length: ", len(slices))
#display.plot_3d(slices, 200)

img = reader.get_image_hu(slices[100])
label, mask, masked_image = transform.lung_mask(img)

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("Input", fontsize=10)
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(label)
plt.title("Clustering", fontsize=10)
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(mask, cmap='gray')
plt.title("Lung Mask", fontsize=10)
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(masked_image, cmap='gray')
plt.title("Masked Image", fontsize=10)
plt.axis('off')

plt.show()
