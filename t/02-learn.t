use strict;
use warnings;

use Test::More tests => 6;
use Test::Number::Delta within => 1e-5;
use Test::Differences;

my $__;
sub NAME { $__ = shift };

###
NAME 'Preparation for the following tests';
require AI::MaxEntropy;
my $me = AI::MaxEntropy->new; 
$me->see(['round', 'smooth', 'red'] => 'apple' => 2);
$me->see(['long', 'smooth', 'yellow'] => 'banana' => 3);
ok $me,
$__;

###
NAME 'Preprocessing';
$me->_preprocess_samples;
eq_or_diff
[
    $me->{ripe_samples},
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
        [ [2, 3, 1] => 0 => 2 ],
	[ [0, 3, 4] => 1 => 3 ]
    ],
    {
        long => 0,
	red => 1,
	round => 2,
	smooth => 3,
	yellow => 4
    },
    {
        apple => 0,
	banana => 1
    },
    [
        'long',
	'red',
	'round',
	'smooth',
	'yellow'
    ],
    [
        'apple',
	'banana'
    ],
    5,
    2,
    10
],
$__;

###
NAME 'Negative log likelihood calculation (lambda = all 0)';
my ($f, $g) = AI::MaxEntropy::_neg_log_likelihood(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], undef, $me
);
delta_ok
[
    $f,
    $g
],
[
    - (2 * log(0.5) + 3 * log(0.5)),
    [
        - (0 * 2 + (0 - 0.5) * 3),
	- ((1 - 0.5) * 2 + 0 * 3),
	- ((1 - 0.5) * 2 + 0 * 3),
	- ((1 - 0.5) * 2 + (0 - 0.5) * 3),
	- (0 * 2 + (0 - 0.5) * 3),
	- (0 * 2 + (1 - 0.5) * 3),
	- ((0 - 0.5) * 2 + 0 * 3),
	- ((0 - 0.5) * 2 + 0 * 3),
	- ((0 - 0.5) * 2 + (1 -0.5) * 3),
	- (0 * 2 + (1 - 0.5) * 3)
    ]
],
$__;

###
NAME 'Negative log likelihood calculation (lambda = random .1 and 0)';
($f, $g) = AI::MaxEntropy::_neg_log_likelihood(
    [.1, .1, 0, 0, 0, .1, .1, 0, 0, .1], undef, $me
);
delta_ok
[
    $f,
    $g
],
[
    - (log(exp(.1) / (2 * exp(.1))) * 2 +
       log(exp(.2) / (exp(.1) + exp(.2))) * 3),
    [
        - (0 * 2 + (0 - exp(.1) / (exp(.1) + exp(.2))) * 3),
	- ((1 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3),
	- ((1 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3),
	- ((1 - exp(.1) / (2 * exp(.1))) * 2 + 
	   (0 - exp(.1) / (exp(.1) + exp(.2))) * 3),
	- (0 * 2 + (0 - exp(.1) / (exp(.1) + exp(.2))) * 3),
	- (0 * 2 + (1 - exp(.2) / (exp(.1) + exp(.2))) * 3),
	- ((0 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3),
	- ((0 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3),
	- ((0 - exp(.1) / (2 * exp(.1))) * 2 +
	   (1 - exp(.2) / (exp(.1) + exp(.2))) * 3),
	- (0 * 2 + (1 - exp(.2) / (exp(.1) + exp(.2))) * 3)
    ]
],
$__;

###
NAME 'Negative log likelihood calculation (with Gaussian smoother)';
$me->{smoother} = { type => 'gaussian', sigma => .5 };
($f, $g) = AI::MaxEntropy::_neg_log_likelihood(
    [.1, .1, 0, 0, 0, .1, .1, 0, 0, .1], undef, $me
);
delta_ok
[
    $f,
    $g
],
[
    - (log(exp(.1) / (2 * exp(.1))) * 2 +
       log(exp(.2) / (exp(.1) + exp(.2))) * 3 -
       (5 * .1 ** 2) / (2 * .5 ** 2)),
    [
        - (0 * 2 + (0 - exp(.1) / (exp(.1) + exp(.2))) * 3 - .1 / .5 ** 2),
	- ((1 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3 - .1 / .5 ** 2),
	- ((1 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3 - 0 / .5 ** 2),
	- ((1 - exp(.1) / (2 * exp(.1))) * 2 + 
	   (0 - exp(.1) / (exp(.1) + exp(.2))) * 3 - 0 / .5 ** 2),
	- (0 * 2 + (0 - exp(.1) / (exp(.1) + exp(.2))) * 3 - 0 / .5 ** 2),
	- (0 * 2 + (1 - exp(.2) / (exp(.1) + exp(.2))) * 3 - .1 / .5 ** 2),
	- ((0 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3 - .1 / .5 ** 2),
	- ((0 - exp(.1) / (2 * exp(.1))) * 2 + 0 * 3 - 0 / .5 ** 2),
	- ((0 - exp(.1) / (2 * exp(.1))) * 2 +
	   (1 - exp(.2) / (exp(.1) + exp(.2))) * 3 - 0 / .5 ** 2),
	- (0 * 2 + (1 - exp(.2) / (exp(.1) + exp(.2))) * 3 - .1 / .5 ** 2)
    ]
],
$__;

###
NAME 'Model object construction';
$me->{smoother} = {};
my $model = $me->learn;
eq_or_diff
[
    $model->{x_bucket},
    $model->{y_bucket},
    $model->{x_list},
    $model->{y_list},
    $model->{x_num},
    $model->{y_num},
    $model->{f_num}
],
[
    {
        long => 0,
	red => 1,
	round => 2,
	smooth => 3,
	yellow => 4
    },
    {
        apple => 0,
	banana => 1
    },
    [
        'long',
	'red',
	'round',
	'smooth',
	'yellow'
    ],
    [
        'apple',
	'banana'
    ],
    5,
    2,
    10
],
$__;

