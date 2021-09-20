import { ImageRelu } from './imageRelu';
import nj from 'numjs';

describe('ImageRelu Layer Test', () => {
  describe('ImageRelu.forwardBatch', () => {
    test('forwardBatch', () => {
      const imageRelu = new ImageRelu();
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
        ],
      ]);
      expect(imageRelu.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
        ],
      ]);
    });
    test('forwardBatch', () => {
      const imageRelu = new ImageRelu();
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
        ],
      ]);
      expect(imageRelu.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
        ],
      ]);
    });
    test('forwardBatch', () => {
      const imageRelu = new ImageRelu();
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
        ],
        [
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
        ],
      ]);
      expect(imageRelu.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
        ],
        [
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
          [
            [1, 5, 5, -0],
            [4, -0, 7, 8],
            [-0, 4, -0, 1],
            [3, -0, 3, 8],
          ],
        ],
      ]);
    });
  });
  describe('ImageRelu.backwardBatch', () => {
    test('backwardBatch', () => {
      const imageRelu = new ImageRelu();
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
        ],
      ]);
      imageRelu.forwardBatch(xBatch);
      const dout = nj.array([
        [
          [
            [1, 5, -5, 6],
            [4, 2, 7, -8],
            [2, 4, 2, 1],
            [-3, -1, 3, 8],
          ],
        ],
      ]);
      expect(imageRelu.backwardBatch(dout).tolist()).toEqual([
        [
          [
            [1, 5, -5, 0],
            [4, 0, 7, -8],
            [0, 4, 0, 1],
            [-3, -0, 3, 8],
          ],
        ],
      ]);
    });
    test('backwardBatch', () => {
      const imageRelu = new ImageRelu();
      const xBatch = nj.array([
        [
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
          [
            [1, 5, 5, -6],
            [4, -2, 7, 8],
            [-2, 4, -2, 1],
            [3, -1, 3, 8],
          ],
        ],
      ]);
      imageRelu.forwardBatch(xBatch);
      const dout = nj.array([
        [
          [
            [1, 5, -5, 6],
            [4, 2, 7, -8],
            [2, 4, 2, 1],
            [-3, -1, 3, 8],
          ],
          [
            [1, 5, -5, 6],
            [4, 2, 7, -8],
            [2, 4, 2, 1],
            [-3, -1, 3, 8],
          ],
        ],
      ]);
      expect(imageRelu.backwardBatch(dout).tolist()).toEqual([
        [
          [
            [1, 5, -5, 0],
            [4, 0, 7, -8],
            [0, 4, 0, 1],
            [-3, -0, 3, 8],
          ],
          [
            [1, 5, -5, 0],
            [4, 0, 7, -8],
            [0, 4, 0, 1],
            [-3, -0, 3, 8],
          ],
        ],
      ]);
    });
  });
});
