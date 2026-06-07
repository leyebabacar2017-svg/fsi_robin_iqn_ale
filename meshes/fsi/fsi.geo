SetFactory("OpenCASCADE");

// =====================================================
// Benchmark Turek-Hron FSI2
// =====================================================

L  = 2.5;
H  = 0.41;

xc = 0.20;
yc = 0.20;
R  = 0.05;

ls = 0.35;
hs = 0.02;

// =====================================================
// Géométrie
// =====================================================

Rectangle(1) = {0,0,0,L,H,0};

Disk(2) = {xc,yc,0,R,R};

Rectangle(3) = {0.25,0.19,0,ls,hs,0};

// =====================================================
// Raffinement structure
// =====================================================

MeshSize{ PointsOf{ Surface{3}; } } = 0.0015;

// =====================================================
// Domaine fluide
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
// Synchronisation
// =====================================================

Coherence;

// =====================================================
// Taille de maille
// =====================================================

Mesh.CharacteristicLengthMin = 0.0025;
Mesh.CharacteristicLengthMax = 0.005;

// =====================================================
// Groupes physiques
// =====================================================

Physical Surface("Fluid",10) = {fluid[0]};
Physical Surface("Solid",20) = {3};

Physical Curve("Cylinder",101) = {5};

Physical Curve("Interface",102) =
{
    6,
    7,
    8
};

Physical Curve("Clamp",103) =
{
    14,
    15
};

Physical Curve("Inlet",104) = {11};

Physical Curve("Outlet",105) = {12};

Physical Curve("BottomWall",106) = {10};

Physical Curve("TopWall",107) = {13};

Field[1] = Distance;
Field[1].SurfacesList = {3};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.0015;
Field[2].LcMax = 0.01;
Field[2].DistMin = 0.01;
Field[2].DistMax = 0.05;

Background Field = 2;