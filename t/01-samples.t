use strict;
use warnings;

use Test::More tests => 6;
use Test::Number::Delta within => 1e-5;

my $__;
sub NAME { $__ = shift };

###
NAME 'Load the module';
BEGIN { use_ok 'AI::MaxEntropy' }

###
NAME 'Create a Maximum Entropy Leaner';
my $me = AI::MaxEntropy->new;
ok $me,
$__;

###
NAME 'Add a sample';
$me->see(['round', 'red'] => 'tomato');
is_deeply
[
    $me->{samples},
    $me->{x_bucket},
    $me->{y_bucket},
    $me->{x_list},
    $me->{y_list},
    $me->{x_num},
    $me->{y_num},
    $me->{f_num}
],
[
    [ [ [ 0, 1 ] => 0 => 1 ] ],
    { round => 0, red => 1 },
    { tomato => 0 },
    [ 'round', 'red' ],
    [ 'tomato' ],
    2,
    1,
    2
],
$__;

###
NAME 'Add more samples';
$me->see(['round', 'smooth', 'red'] => 'apple' => 2);
$me->see(['long', 'smooth', 'yellow'] => 'banana' => 3);
is_deeply 
[
    $me->{samples},
    $me->{x_bucket},
    $me->{y_bucket},
    $me->{x_list},
    $me->{y_list},
    $me->{x_num},
    $me->{y_num},
    $me->{f_num}
],
[
    [
        [ [ 0, 1 ] => 0 => 1 ],
	[ [ 0, 2, 1 ] => 1 => 2 ],
	[ [ 3, 2, 4 ] => 2 => 3 ]
    ],
    { round => 0, red => 1, smooth => 2, long => 3, yellow => 4 },
    { tomato => 0, apple => 1, banana => 2 },
    [ 'round', 'red', 'smooth', 'long', 'yellow' ],
    [ 'tomato', 'apple', 'banana' ],
    5,
    3,
    15
],
$__;

###
NAME 'Forget samples';
$me->forget_all;
is_deeply 
[
    $me->{samples},
    $me->{x_bucket},
    $me->{y_bucket},
    $me->{x_list},
    $me->{y_list},
    $me->{x_num},
    $me->{y_num},
    $me->{f_num}
],
[
    [],
    {},
    {},
    [],
    [],
    0,
    0,
    0
],
$__;

###
NAME 'Add attribute-value samples';
$me->see({color => ['red', 'green'], shape => 'round'} => 'apple');
$me->see({surface => 'smooth', color => 'red'} => 'tomato');
is_deeply
[
    $me->{samples},
    $me->{x_bucket},
    $me->{y_bucket},
    $me->{x_list},
    $me->{y_list},
    $me->{x_num},
    $me->{y_num},
    $me->{f_num}
],
[
    [
        [ [ 0, 1, 2 ] => 0 => 1 ],
	[ [ 3, 0 ] => 1 => 1 ]
    ],
    { 'color:red' => 0, 'color:green' => 1, 
      'shape:round' => 2, 'surface:smooth' => 3 },
    { 'apple' => 0, 'tomato' => 1 },
    [ 'color:red', 'color:green', 'shape:round', 'surface:smooth' ],
    [ 'apple', 'tomato' ],
    4,
    2,
    8
],
$__;

