import { Layer } from './base';
import nj from 'numjs';

export class Relu implements Layer {
  x: nj.NdArray<number>;
  xBatch: nj.NdArray<number[]>;

  constructor(x?: nj.NdArray<number>, xBatch?: nj.NdArray<number[]>) {
    this.x = x == null ? nj.zeros(0) : x;
    this.xBatch = xBatch == null ? nj.zeros(0) : xBatch;
  }

  forward = (x: nj.NdArray<number>): nj.NdArray<number> => {
    this.x = x;
    const xArray = x.tolist();
    return nj.array(xArray.map((x) => (x > 0 ? x : 0)));
  };

  forwardBatch = (xBatch: nj.NdArray<number[]>): nj.NdArray<number[]> => {
    this.xBatch = xBatch;
    const xArray = xBatch.tolist();
    return nj.array(
      xArray.map((xArr: number[]) => {
        return xArr.map((x) => (x > 0 ? x : 0));
      })
    );
  };

  backward = (): nj.NdArray<number> => {
    const xArray = this.x.tolist();
    return nj.array(xArray.map((x) => Number(x > 0)));
  };

  backwardBatch = (): nj.NdArray<number[]> => {
    const xArrayBatch = this.xBatch.tolist();
    return nj.array(
      xArrayBatch.map((xArray) => xArray.map((x) => Number(x > 0)))
    );
  };
}
