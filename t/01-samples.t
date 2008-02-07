use strict;
use warnings;

use Test::More tests => 6;
use Test::Number::Delta within => 1e-5;
use Test::Differences;

my $__;
sub NAME { $__ = shift };

###
NAME 'Load the Module';
use_ok 'AI::MaxEntropy',
$__;

###
NAME 'Create a Maximum Entropy Leaner';
my $me = AI::MaxEntropy->new;
ok $me,
$__;

###
NAME 'Add a sample';
$me->see(['round', 'red'] => 'tomato');
eq_or_diff [$me->{samples}, $me->{x_bucket}, $me->{y_bucket}],
[
    [ [ ['round', 'red'] => 'tomato' => 1 ] ],
    { red => undef, round => undef },
    { tomato => undef }
],
$__;

###
NAME 'Forget one sample';
$me->forget_all;
eq_or_diff [$me->{samples}, $me->{x_bucket}, $me->{y_bucket}],
[
    [], {}, {}
],
$__;

###
NAME 'Add multiple samples';
$me->see(['round', 'smooth', 'red'] => 'apple' => 2);
$me->see(['long', 'smooth', 'yellow'] => 'banana' => 3);
eq_or_diff [$me->{samples}, $me->{x_bucket}, $me->{y_bucket}],
[
    [
        [ ['round', 'smooth', 'red'] => 'apple' => 2 ],
	[ ['long', 'smooth', 'yellow'] => 'banana' => 3]
    ],
    {
        long => undef,
	red => undef,
	round => undef,
	smooth => undef,
	yellow => undef
    },
    {
        apple => undef,
	banana => undef
    }
],
$__;

###
NAME 'Forget multiple samples';
$me->forget_all;
eq_or_diff [$me->{samples}, $me->{x_bucket}, $me->{y_bucket}],
[
    [], {}, {}
],
$__;

