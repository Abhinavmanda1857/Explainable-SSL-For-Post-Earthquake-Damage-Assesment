**Building Damage Segmentation – Streamlit Inference App**

This Streamlit application performs building damage segmentation using a Vision Transformer (ViT-Tiny) model trained on pre- and post-disaster satellite images.
The model predicts four classes of damage: no-damage, minor, major, and destroyed.

**Features**
* Upload pre-disaster and post-disaster images
* Optional: use post image as pre if only one image is available
* Runs segmentation using a ViT-Tiny model adapted for 6-channel input

**Displays**
- Predicted class mask
- Color-coded categorical map
- Overlay of predictions on the post image
- Counts buildings using connected components
- Allows downloading the predicted mask
- Optional IoU calculation if ground truth mask is provided
  

**Classes**
- 0	No-damage
- 1	Minor
- 2	Major
- 3	Destroyed
  


  
