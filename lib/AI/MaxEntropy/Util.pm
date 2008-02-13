use strict;
use warnings;

package AI::MaxEntropy::Util;

use Exporter;

our $VERSION = '0.10';

our @ISA = qw/Exporter/;

our @EXPORT_OK = qw/train_and_test precision recall/;
our %EXPORT_TAGS = (all => [@EXPORT_OK]);

sub train_and_test {
    my ($me, $samples, $map) = @_;
    my $p = length($map);
    my $n = scalar(@$samples);
    # collect training samples
    for my $i (0 .. $p - 1) {
        if (substr($map, $i, 1) eq 'x') {
	    $me->see(@{$samples->[$_]})
	        for (int($n * $i / $p) .. int($n * ($i + 1) / $p) - 1);
	}
    }
    # do training
    my $m = $me->learn;
    # do testing
    my $r = [];
    for my $i (0 .. $p - 1) {
        if (substr($map, $i, 1) eq 'o') {
	    push @$r, [$samples->[$_] => $m->predict($samples->[$_]->[0])]
	        for (int($n * $i / $p) .. int($n * ($i + 1) / $p) - 1);
	}
    }
    return ($r, $m);
}

sub precision {
    my $r = shift;
    my ($c, $n) = (0, 0);
    for (@$r) {
        my $w = defined($_->[0]->[2]) ? $_->[0]->[2] : 1;
        $n += $w;
        $c += $w if $_->[0]->[1] eq $_->[1];
    }
    return $c / $n;
}

sub recall {
    my $r = shift;
    my $label = shift;
    my ($c, $n) = (0, 0);
    for (@$r) {
        if ($_->[0]->[1] eq $label) {
            my $w = defined($_->[0]->[2]) ? $_->[0]->[2] : 1;
	    $n += $w;
	    $c += $w if $_->[1] eq $label;
	}
    }
    return $c / $n;
}

1;

__END__

=head1 NAME

AI::MaxEntropy::Util - Utilities for doing experiments with ME learners 

=head1 SYNOPSIS

  use AI::MaxEntropy;
  use AI::MaxEntropy::Util qw/:all/;

  my $me = AI::MaxEntropy->new;
  
  my $samples = [
      [['a', 'b', 'c'] => 'x'],
      [['e', 'f'] => 'y' => 1.5],
      ...
  ];

  my ($result, $model) = train_and_test($me, $samples, 'xxxo');

  print precision($result)."\n";
  print recall($result, 'x')."\n";

=head1 DESCRIPTION

This module makes doing experiments with Maximum Entropy learner easier.

Generally, an experiment involves a training set and a testing set
(sometimes also a parameter adjusting set). The learner is trained with
samples in the training set and tested with samples in the testing set.
Usually, 2 measures of performance are concerned.
One is precision, indicating the percentage of samples which are correctly
predicted in the testing set. The other one is recall, indicating the 
precision of samples with a certain label.

=head1 FUNCTIONS

=head2 train_and_test

This function automated the process of training and testing.

    my $me = AI::MaxEntropy->new;
    my $sample = [
        [ ['a', 'b'] => 'x' => 1.5 ],
	...
    ];
    my ($result, $model) = train_and_test($me, $sample, 'xxxo');

First, the whole samples set will be divided into a training set and a
testing set according to the specified pattern. A pattern is a string,
in which each character stands for a part of the samples set.
If the character is C<'x'>, the corresponding part is used for training.
If the character is C<'o'>, the corresponding part is used for testing.
Otherwise, the corresponding part is simply ignored.

For example, the pattern 'xxxo' means the first three forth of the samples
set are used for training while the last one forth is used for testing.

The function returns two values. The first one is an array ref describe
the result of testing, in which each element follows a structure like
C<[sample =E<gt> result]>. The second one is the model learnt from the
training set, which is an L<AI::MaxEntropy::Model> object.

=head2 precision

Calculates the precision based on the result returned by
L</train_and_test>.

  ...
  my ($result, $model) = train_and_test(...);
  print precision($result)."\n";

Note that the weights of samples are taken into consideration.

=head2 recall

Calculates the recall of a certain label based on the result returned by
L</train_and_test>.

  ...
  my ($result, $model) = train_and_test(...);
  print recall($result, 'label')."\n";

Note that the weights of samples are taken into consideration.

=head1 SEE ALSO

L<AI::MaxEntropy>, L<AI::MaxEntropy::Model>

=head1 AUTHOR

Laye Suen, E<lt>laye@cpan.orgE<gt>

=head1 COPYRIGHT AND LICENSE

The MIT License

Copyright (C) 2008, Laye Suen

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

