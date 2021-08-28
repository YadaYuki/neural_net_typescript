import { Relu } from './relu';

describe('Affine Layer Test', () => {
  describe('Affine.forward', () => {
    test('forward', () => {
      const relu = new Relu();
      const x = nj.array([-1, 1, 2, 3]);
      expect(relu.forward(x).tolist()).toEqual([0, 1, 2, 3]);
    });
  });
});
