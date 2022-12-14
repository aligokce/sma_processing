from .em32 import EigenmikeEM32

microphones = {
    'em32': EigenmikeEM32().returnAsStruct(),
    'Eigen': EigenmikeEM32().returnAsStruct(),
}