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
  });
});
