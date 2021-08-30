import { Layer } from './base';
import nj from 'numjs';
import { softmax } from '../utils/activation';
import { crossEntroyError } from '../utils/loss';

export class SoftmaxWithLoss implements Layer {
  y: nj.NdArray<number>;
  yBatch: nj.NdArray<number[]>;
  t: nj.NdArray<number>;
  tBatch: nj.NdArray<number[]>;
  loss: number;
  constructor() {
    this.y = nj.zeros(0);
    this.yBatch = nj.zeros(0);
    this.t = nj.zeros(0);
    this.tBatch = nj.zeros(0);
    this.loss = -1;
  }
  forward(x: nj.NdArray<number>, t: nj.NdArray<number>): number {
    this.y = softmax(x);
    this.t = t;
    this.loss = crossEntroyError(this.y, this.t);
    return this.loss;
  }
  backward(): void {
    return;
  }
}
