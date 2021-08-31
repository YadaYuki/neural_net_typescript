import { Layer } from './base';
import nj from 'numjs';

export class Relu implements Layer {
  mask: nj.NdArray<number>;
  maskBatch: nj.NdArray<number[]>;

  constructor(
    mask: nj.NdArray<number> = nj.zeros(0),
    maskBatch: nj.NdArray<number[]> = nj.zeros(0)
  ) {
    this.mask = mask;
    this.maskBatch = maskBatch;
  }

  forward = (x: nj.NdArray<number>): nj.NdArray<number> => {
    const xArray = x.tolist();
    this.mask = nj.array(xArray.map((xItem) => Number(xItem > 0)));
    return x.multiply(this.mask);
  };

  forwardBatch = (xBatch: nj.NdArray<number[]>): nj.NdArray<number[]> => {
    const xArrayBatch = xBatch.tolist();
    this.maskBatch = nj.array(
      xArrayBatch.map((xArray) => xArray.map((x) => Number(x > 0)))
    );
    return xBatch.multiply(this.maskBatch);
  };

  backward = (dout: nj.NdArray<number>): nj.NdArray<number> => {
    return dout.multiply(this.mask);
  };

  backwardBatch = (dout: nj.NdArray<number[]>): nj.NdArray<number[]> => {
    return dout.multiply(this.maskBatch);
  };
}
