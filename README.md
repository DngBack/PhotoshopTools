# Using TRACER and Stable Diffusion to changing background in image with prompt

Update Task :

- Background Diffusion
- Background Removal
- Remove Object with Inpainting
- Inpainting image

## Background Changing Example

|Order | Text Prompt | Image Imput | Output Image | Size Image (width, height) | Step |
|------|-------------|-------------|--------------|------------|-------|
| 1 | A lady standing in Office of Technology Company, a bright office job. | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out1.png)|  (500, 750) | 20 |
| 2 | A lady standing before The warm Coffee House with some plants , some lamp and a big bookself | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out5.png)| (500, 750) | 20 |
| 3 | Red Sofa | ![Image](BackGroundChanging/Image/Test4.jpg) | ![Image](BackGroundChanging/Image/Test4Out.png)| (500, 750) | 20 |
| 4 | A young woman is lying on a blue sofa, next to a table with a lamp and some books, in a shinesine house. This house have bookself. | ![Image](BackGroundChanging/Image/Test4.jpg) | ![Image](BackGroundChanging/Image/Test4Out2.png)| (500, 750) | 20 |
| 5 | A Coffe House | ![Image](BackGroundChanging/Image/Test3.jpg) | ![Image](BackGroundChanging/Image/Test3Out.png)| (640, 360) | 20 |
| 6 | A man is standing in a pedestrian street with lots of trees and lots of sunlight. | ![Image](BackGroundChanging/Image/Test3.jpg) | ![Image](BackGroundChanging/Image/Test3Out1.png)| (640, 360) | 20 |
| 7 | The man is standing in front of a cafe with a few tall trees and a bus stop | ![Image](BackGroundChanging/Image/Test2.png) | ![Image](BackGroundChanging/Image/Test2Out2.png)| (960, 550) | 20 |

## Background Removal Example

|Order | Image Imput | Output Image |
|------|-------------|--------------|
| 1 | ![Image](BackgroundRemoval/Image/Test2.png) | ![Image](BackgroundRemoval/Image/Output1.png) |
