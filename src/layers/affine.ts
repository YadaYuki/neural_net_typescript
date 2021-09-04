import { Layer } from './base';
import nj from 'numjs';

export class Affine implements Layer {
  W: nj.NdArray<number[]>;
  b: nj.NdArray<number>;
  dW: nj.NdArray<number[]>;
  db: nj.NdArray<number>;
  x: nj.NdArray<number>;
  xBatch: nj.NdArray<number[]>;
  constructor(
    W: nj.NdArray<number[]>,
    b: nj.NdArray<number>,
    x: nj.NdArray<number> = nj.zeros(0),
    xBatch: nj.NdArray<number[]> = nj.zeros(0)
  ) {
    this.W = W;
    this.b = b;
    this.dW = nj.zeros(0);
    this.db = nj.zeros(0);
    this.x = x;
    this.xBatch = xBatch;
  }

  /*
    forwardは X・W + b を返す。
  */
  forward(x: nj.NdArray<number>): nj.NdArray<number> {
    this.x = x;
    const xMat = x.reshape(1, x.size) as nj.NdArray<number[]>;
    const bMat = this.b.reshape(1, this.b.size) as nj.NdArray<number[]>;
    return nj.add(nj.dot(xMat, this.W), bMat).flatten();
  }

  forwardBatch(xBatch: nj.NdArray<number[]>): nj.NdArray<number[]> {
    this.xBatch = xBatch;
    const batchSize = xBatch.shape[0];
    const bMat = this.b.reshape(1, this.b.size) as nj.NdArray<number[]>;
    const ones = nj.ones(batchSize).reshape(batchSize, 1) as nj.NdArray<
      number[]
    >;
    const bMatAdd = nj.dot(ones, bMat);
    return nj.dot(xBatch, this.W).add(bMatAdd);
  }

  backward(dout: nj.NdArray<number>): nj.NdArray<number> {
    this.db = dout;
    this.dW = nj.dot(
      this.x.reshape(this.x.size, 1) as nj.NdArray<number[]>,
      dout.reshape(1, dout.size) as nj.NdArray<number[]>
    );
    return nj
      .dot(dout.reshape(1, dout.size) as nj.NdArray<number[]>, this.W.T)
      .flatten();
  }

  backwardBatch(dout: nj.NdArray<number[]>): nj.NdArray<number[]> {
    const batchSize = this.xBatch.shape[0];
    this.db = nj
      .dot(
        dout.T,
        nj.ones(batchSize).reshape(batchSize, 1) as nj.NdArray<number[]>
      )
      .flatten();
    this.dW = nj.dot(this.xBatch.T, dout);
    return nj.dot(dout, this.W.T);
  }
}
