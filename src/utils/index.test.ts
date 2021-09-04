import { maxIdx } from './index';

describe('index.ts test', () => {
  describe('maxIdx', () => {
    test('maxIdx([1, 2, 3, 4, 5]) should be 4', () => {
      expect(maxIdx([1, 2, 3, 4, 5])).toBe(4);
    });
    test('maxIdx', () => {
      expect(maxIdx([4, 2, 5, 7, 1, 3, 5, 22, 24, 1, 1, 3])).toBe(8);
    });
  });
});
