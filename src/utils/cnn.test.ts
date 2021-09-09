import { im2col } from './cnn';
import nj from 'numjs';

describe('utils/cnn test', () => {
  describe('im2col', () => {
    test('', () => {
      const inputData = nj.array([
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
          [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
          ],
        ],
      ]);
      const col = im2col(inputData, 2, 2);
      expect(col.tolist()).toEqual([
        [1, 2, 4, 5, 10, 11, 13, 14],
        [2, 3, 5, 6, 11, 12, 14, 15],
        [4, 5, 7, 8, 13, 14, 16, 17],
        [5, 6, 8, 9, 14, 15, 17, 18],
      ]);
    });
    test('', () => {
      const inputData = nj.array([
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
          [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
          ],
        ],
      ]);
      const col = im2col(inputData, 3, 3);
      expect(col.tolist()).toEqual([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
      ]);
    });
  });
});
