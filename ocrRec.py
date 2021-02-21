import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr_for_single_line(img)
print("Predicted Chars:", res)
