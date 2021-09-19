import { Pooling } from './pooling';
import nj from 'numjs';

describe('Pooling Layer Test', () => {
  describe('Pooling.forward', () => {
    test('forward', () => {
      const pooling = new Pooling(2, 2);
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, 6],
            [4, 3, 7, 8],
            [2, 4, 2, 1],
            [3, 1, 3, 8],
          ],
        ],
      ]);
      expect(pooling.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [5, 7, 8],
            [4, 7, 8],
            [4, 4, 8],
          ],
        ],
      ]);
    });
    test('forward', () => {
      const pooling = new Pooling(2, 2, 2, 0);
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, 6],
            [4, 3, 7, 8],
            [2, 4, 2, 1],
            [3, 1, 3, 8],
          ],
        ],
      ]);
      expect(pooling.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [5, 8],
            [4, 8],
          ],
        ],
      ]);
    });
  });
  describe('Pooling.forward', () => {
    test('backward', () => {
      const pooling = new Pooling(2, 2);
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, 6],
            [4, 3, 7, 8],
            [2, 4, 2, 1],
            [3, 1, 3, 8],
          ],
        ],
      ]);
      pooling.forwardBatch(xBatch);
      const dout = nj.ones([1, 1, 3, 3]).reshape(1, 1, 3, 3) as nj.NdArray<
        number[][][]
      >;
      expect(pooling.backwardBatch(dout).tolist()).toEqual([
        [
          [
            [0, 1, 0, 0],
            [1, 0, 2, 2],
            [0, 2, 0, 0],
            [0, 0, 0, 1],
          ],
        ],
      ]);
    });
  });
  test('backward', () => {
    const pooling = new Pooling(2, 2);
    const xBatch = nj.array([
      [
        [
          [1, 5, 5, 6],
          [4, 3, 7, 8],
          [2, 4, 2, 1],
          [3, 1, 3, 8],
        ],
      ],
    ]);
    pooling.forwardBatch(xBatch);
    const dout = nj.arange(9).reshape(1, 1, 3, 3) as nj.NdArray<number[][][]>;
    expect(pooling.backwardBatch(dout).tolist()).toEqual([
      [
        [
          [0, 0, 0, 0],
          [3, 0, 5, 7],
          [0, 13, 0, 0],
          [0, 0, 0, 8],
        ],
      ],
    ]);
  });
});
