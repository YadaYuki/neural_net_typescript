export interface FixedLengthArray<L extends number, T> extends ArrayLike<T> {
  length: L;
}
