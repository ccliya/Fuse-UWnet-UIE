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
    
    
    
