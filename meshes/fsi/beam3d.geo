SetFactory("OpenCASCADE");

Rectangle(1) =
{
    0.25, 0.19, 0,
    0.35, 0.02
};

Extrude {0,0,0.01}
{
    Surface{1};
}

Mesh.CharacteristicLengthMin = 0.002;
Mesh.CharacteristicLengthMax = 0.002;
