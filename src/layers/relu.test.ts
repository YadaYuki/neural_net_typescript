import { Relu } from './relu';
import nj from 'numjs';

describe('Relu Layer Test', () => {
  describe('Relu.forward', () => {
    test('', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, 2, 3]);
      expect(relu.forward(x).tolist()).toEqual([0, 1, 2, 3]);
    });
    test('', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, -2, 0]);
      expect(relu.forward(x).tolist()).toEqual([0, 3, 0, 0]);
    });
  });
});
