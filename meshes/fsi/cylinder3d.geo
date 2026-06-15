SetFactory("OpenCASCADE");

Cylinder(1) =
{
  0.20, 0.20, 0.0,
  0.0,  0.0,  0.01,
  0.05
};

Mesh.CharacteristicLengthMin = 0.003;
Mesh.CharacteristicLengthMax = 0.003;

// surfaces du cylindre
Physical Surface("CylinderSurface") = {1,2,3};
