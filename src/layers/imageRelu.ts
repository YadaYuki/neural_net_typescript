import { Layer } from './base';
import nj from 'numjs';

/*
 画像などの3次元データのためのRelu関数を表すクラス。ConvolutionとPoolingの間に入る層として用いる。
*/
export class ImageRelu implements Layer {
  maskBatch: nj.NdArray<number[][][]> = nj.zeros(0);

  forward(): void {
    return;
  }
  forwardBatch(xBatch: nj.NdArray<number[][][]>): nj.NdArray<number[][][]> {
    const xImageArray = xBatch.tolist();
    this.maskBatch = nj.array(
      xImageArray.map((xImage) =>
        xImage.map((xChannel) =>
          xChannel.map((xArray) => xArray.map((x) => Number(x > 0)))
        )
      )
    );
    return xBatch.multiply(this.maskBatch);
  }

  backward(): void {
    return;
  }

  backwardBatch(dout: nj.NdArray<number[][][]>): nj.NdArray<number[][][]> {
    return dout.multiply(this.maskBatch);
  }
}
