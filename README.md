# Fuse-UWnet-UIE
base on shallow-Uwnet and find a better UIE
主要分为浅层、中层和深层三层图像增强

as following：
  
  #shallow layer
    
    1、
    RGB → Conv2d → Max pooling → flatten → Mamba Block → reshape → ReLU +Conv → o0
    Lab → Conv2d → Max pooling → flatten → Mamba Block → reshape → ReLU +Conv → oI0
    2、  o0+oI0 → AdaIn + DenSoA（reference: https://blog.csdn.net/adventure_man/article/details/140588342）
    3、AdaIn_DenSoA(o0,oI0) → ReLU+Conv → shalllow feature_vector

  #middle layer
    
    1、 
    RGB → ConvBlock+ELA+DeConv → om
    lab → ConvBlock+ELA+DeConv → olm
    2、AdaIn_DenSoA(o0,oI0) → ReLU+Conv → AdaIn_DenSoA(om,olm) → mid feature_vector

  #deep layer
   
    1、
    RGB → ConvBlock+ELA+DeConv → od
    lab → ConvBlock+ELA+DeConv → old
    2、AdaIn_DenSoA(om,olm) → → ReLU+Conv → AdaIn_DenSoA(od,old) → final feature_vector

  #output
    
    final feature_vector → RGB enhancement



    ===================above are date 2025.07.18====================================
    result: the above methods had week score of standard UIQM\SSIM\PSNR(epoch1-100):
    指标	        MIN	      MAX	      STABLE	      CONCLUSION
    Avg Loss      0.2709	  0.0099	  < 0.02	      模型已充分收敛
    UIQM	        0.0706	  0.4020	  ~0.30–0.38	  增强质量中等，有所提升
    SSIM	        0.2435	  0.4060	  ~0.36–0.39	  图像结构恢复中等
    PSNR	        10.52 dB	12.73 dB	~11.6–12.6 dB	像素保真度较低

    THE NEXT STEP IS consider as the following:
    1、try the optimized _perceptual loss_ represent or help the MSE。 
    2、lead in _UIQM-aware_ or _SSIM-aware_ training mechanism.
    3、try to use UGAN to enhance the image reality texture.
    4、focus on the pair strategy of the clean-fuse of datasets.
    ================================================================================
