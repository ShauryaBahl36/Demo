from flask import Flask, render_template, request
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np

app = Flask(__name__)

constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100
constant_parameter_4 = 18

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image_file']
        if image_file:
            # Save the uploaded image file
            image_path = './static/uploaded_image.jpg'
            image_file.save(image_path)

            # Extract the signature from the image
            signature_path = extract_signature(image_path)

            return render_template('index.html', image_path=image_path, signature_path=signature_path)

    return render_template('index.html')

def extract_signature(img_path):
    try:
        # Read the input image
        img = cv2.imread(img_path, 0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary

        if img is not None:
            # Connected component analysis using scikit-learn framework
            blobs = img > img.mean()
            blobs_labels = measure.label(blobs, background=1)
            image_label_overlay = label2rgb(blobs_labels, image=img)

            the_biggest_component = 0
            total_area = 0
            counter = 0
            average = 0.0
            for region in regionprops(blobs_labels):
                if region.area > 10:
                    total_area += region.area
                    counter += 1
                if region.area >= 250:
                    if region.area > the_biggest_component:
                        the_biggest_component = region.area

            average = total_area / counter

            # Calculate threshold values
            a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
            a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

            # Remove small and large connected pixels
            pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
            component_sizes = np.bincount(pre_version.ravel())
            too_small = component_sizes > a4_big_size_outliar_constant
            too_small_mask = too_small[pre_version]
            pre_version[too_small_mask] = 0

            # Save the pre-version image as the signature
            signature_path = './static/signature.jpg'
            plt.imsave(signature_path, pre_version, cmap='gray')

            return signature_path
        else:
            return None
    except:
        return None

if __name__ == '__main__':
    app.run(debug=True)
