SetFactory("OpenCASCADE");

// =====================================================
// Turek-Hron FSI2
// OpenFOAM mesh
// =====================================================

L  = 2.5;
H  = 0.41;

xc = 0.20;
yc = 0.20;
R  = 0.05;

ls = 0.35;
hs = 0.02;

thickness = 0.01;

// =====================================================
// Geometry
// =====================================================

Rectangle(1) = {0,0,0,L,H};

Disk(2) = {xc,yc,0,R,R};

Rectangle(3) = {0.25,0.19,0,ls,hs};

// =====================================================
// Fluid domain
// =====================================================

fluid[] = BooleanDifference
{
  Surface{1};
  Delete;
}
{
  Surface{2};
  Surface{3};
};

// =====================================================
// Mesh sizes
// =====================================================

Mesh.CharacteristicLengthMin = 0.0025;
Mesh.CharacteristicLengthMax = 0.005;

Field[1] = Distance;
Field[1].SurfacesList = {3};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.0015;
Field[2].LcMax = 0.01;
Field[2].DistMin = 0.01;
Field[2].DistMax = 0.05;

Background Field = 2;

// =====================================================
// Extrusion
// =====================================================

out[] = Extrude {0,0,thickness}
{
  Surface{fluid[0]};
  Layers{1};
  Recombine;
};

// =====================================================
// Diagnostic
// =====================================================

Printf("Top surface = %g", out[0]);
Printf("Volume      = %g", out[1]);

For i In {2:#out[]-1}
  Printf("Lateral surface = %g", out[i]);
EndFor

// =====================================================
// Physical volume
// =====================================================

Physical Volume("FluidVolume",1000) = {out[1]};

// =====================================================
// Front / Back
// =====================================================

Physical Surface("Front",200) = {fluid[0]};
Physical Surface("Back",201)  = {out[0]};

