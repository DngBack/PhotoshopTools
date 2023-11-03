# Using TRACER and Stable Diffusion to changing background in image with prompt

Update Task :

- Background Diffusion
- Background Removal
- Remove Object with Inpainting

## Background Diffusion

|Order | Text Prompt | Image Imput | Output Image | Size Image (width, height) | Step |
|------|-------------|-------------|--------------|------------|-------|
| 1 | Office in Maketing Company | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out.png)| (500, 750) | 20 |
| 2 | A lady standing in Office of Technology Company, a bright office job. | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out1.png)|  (500, 750) | 20 |
| 3 | Coffe House | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out2.png)| (500, 750) | 20 |
| 4 | Coffe House, with Lots of sunshine, some plants and bright photos | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out3.png)| (500, 750) | 20 |
| 5 | A lady standing before Coffe House, with Lots of sunshine, some plants and bright photos | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out4.png)| (500, 750) | 20 |
| 6 | A lady standing before The warm Coffee House with some plants , some lamp and a big bookself | ![Image](BackGroundChanging/Image/Test1.jpg) | ![Image](BackGroundChanging/Image/Test1Out5.png)| (500, 750) | 20 |
| 7 | Red Sofa | ![Image](BackGroundChanging/Image/Test4.jpg) | ![Image](BackGroundChanging/Image/Test4Out.png)| (500, 750) | 20 |
| 8 | Lying on Red Sofa, beside table with lamp and some book | ![Image](BackGroundChanging/Image/Test4.jpg) | ![Image](BackGroundChanging/Image/Test4Out1.png)| (500, 750) | 20 |
| 9 | A young woman is lying on a blue sofa, next to a table with a lamp and some books, in a shinesine house. This house have bookself. | ![Image](BackGroundChanging/Image/Test4.jpg) | ![Image](BackGroundChanging/Image/Test4Out2.png)| (500, 750) | 20 |
| 10 | A Coffe House | ![Image](BackGroundChanging/Image/Test3.jpg) | ![Image](BackGroundChanging/Image/Test3Out.png)| (640, 360) | 20 |
| 11 | A man is standing in a pedestrian street with lots of trees and lots of sunlight. | ![Image](BackGroundChanging/Image/Test3.jpg) | ![Image](BackGroundChanging/Image/Test3Out1.png)| (640, 360) | 20 |
| 12 | A Coffe House | ![Image](BackGroundChanging/Image/Test2.png) | ![Image](BackGroundChanging/Image/Test2OutOut.png)| (960, 550) | 20 |
| 12 | The man is standing in front of a cafe with a few tall trees and a bus stop | ![Image](BackGroundChanging/Image/Test2.png) | ![Image](BackGroundChanging/Image/Test2Out2.png)| (960, 550) | 20 |
| 12 | The man is standing in front of a cafe with a few tall trees and a bus stop | ![Image](BackGroundChanging/Image/Test2.png) | ![Image](BackGroundChanging/Image/Test2Out3.png)| (960, 550) | 30 |
