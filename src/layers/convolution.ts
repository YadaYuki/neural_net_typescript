import { Layer } from './base';
import nj from 'numjs';

export class Convolution implements Layer {
  W: nj.NdArray<number[][][]>;
  b: nj.NdArray<number>;
  stride: number;
  pad: number;
  dW: nj.NdArray<number[][][]>;
  db: nj.NdArray<number>;
  // for backward
  x: nj.NdArray<number[][][]>;
  colX: nj.NdArray<number[]>;
  colW: nj.NdArray<number[]>;
  constructor(
    W: nj.NdArray<number[][][]>,
    b: nj.NdArray<number>,
    stride = 1,
    pad = 0
  ) {
    this.W = W;
    this.b = b;
    this.stride = stride;
    this.pad = pad;
    this.dW = nj.zeros(0);
    this.db = nj.zeros(0);
    this.x = nj.zeros(0);
    this.colX = nj.zeros(0);
    this.colW = nj.zeros(0);
  }
  forward(): void {
    return;
  }
  forwardBatch(xBatch: nj.NdArray<number[][][]>): void {
    return;
  }
  backward(): void {
    return;
  }
  backwardBatch(): void {
    return;
  }
}
