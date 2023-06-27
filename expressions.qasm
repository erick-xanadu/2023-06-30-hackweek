
gate h q {
  U(1.57079, 0.0, 3.141592) q;
  gphase(-0.78539);
}

gate X q {
  U(3.14159, 0.0, 3.14159) q;
  gphase(-1.57079);
}

gate rx(theta) q {
  U(theta, -1.57079, 1.57079) q;
  gphase(-theta / 2.0);
}

qubit x;
rx(3.14159) x;
