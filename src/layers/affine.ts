import { Layer } from './base';
import nj from 'numjs';

export class Affine implements Layer {
  W: nj.NdArray<number[]>;
  b: nj.NdArray<number>;
  dW: nj.NdArray<number[]>;
  db: nj.NdArray<number>;
  x: nj.NdArray<number>;
  xBatch: nj.NdArray<number[]>;
  constructor(W: nj.NdArray<number[]>, b: nj.NdArray<number>) {
    this.W = W;
    this.b = b;
    this.dW = nj.zeros(0);
    this.db = nj.zeros(0);
    this.x = nj.zeros(0);
    this.xBatch = nj.zeros(0);
  }
  forward(): void {
    1 + 1;
  }
  forwardBatch(): void {
    1 + 1;
  }

  backward(): void {
    1 + 1;
  }
  backwardBatch(): void {
    1 + 1;
  }
}
