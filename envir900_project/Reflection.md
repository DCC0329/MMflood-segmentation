## Reflection

### 1. What I learned about the dataset
Working on this project helped me understand how messy flood segmentation with SAR imagery actually is, and why the modeling choices ended up mattering more than I expected. The first thing that stood out is just how imbalanced the MMFlood dataset is. Most tiles barely contain any flooded pixels, and some have almost none. When I trained the first baseline model without addressing this imbalance, the network basically learned to predict “no flood” everywhere. The recall was extremely low, and the predictions looked almost empty. Once I added weighted sampling and a small mask ratio (2%), the model finally started paying attention to flooded areas.

### 2. What I learned about the models
I also learned that using a deeper backbone doesn’t automatically lead to better performance. I initially assumed ResNet101 would beat ResNet50, but it didn’t. On this dataset, R101 tended to overfit the background and didn’t actually capture more meaningful features from SAR. A medium-sized backbone turned out to be a more reasonable choice.

Adding DEM wasn’t very helpful either. I only used DEM as an extra channel, and since the model processes it in the same encoder as SAR, it likely couldn’t learn any terrain-specific patterns. This made me realize that DEM probably needs a more thoughtful fusion strategy instead of just stacking channels together.

### 3. What didn’t work and why
Several design choices that seemed promising in theory did not translate into better performance.  
- The deeper backbone overfit background noise instead of improving feature extraction.  
- Naive multimodal fusion (SAR + DEM stacked as channels) did not provide any benefit.  
- The baseline model without imbalance handling collapsed completely, showing how strongly the dataset properties dictate model behavior.  

These failures helped clarify which components actually matter in SAR flood mapping.

### 4. What I would try next
If I kept working on this, I would probably look into transformer-based architectures, since they handle global context better, and maybe try a multi-encoder design so SAR and DEM can be learned separately. I would also want to test different sampling strategies or even dynamic mask ratios to deal with the imbalance more systematically.

### 5. Overall reflection
Through all the experiments, debugging, and visualization, I feel I gained a much more practical understanding of what actually matters for SAR-based flood mapping. I’m also clearer about which directions might be worth trying in the future. Although the project was challenging at times, the process itself helped me build more confidence in working with SAR data and deep learning models.
