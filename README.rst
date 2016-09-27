==========================
PB based random attributes
==========================


This code generates data for tables that can be used to generate "random" attributes.

Original idea from reddit: https://www.reddit.com/r/dndnext/comments/54ext0/rolling_for_point_buy/

What have been changed
======================

Original tables treated all stats the same: chance to get ``13 13 13 12 12 12``
is the same as chance to get ``16 15 13 8 8 8``.

But if you a rolling stats with (4d6 keep 3 largest) then you'll have  about 65 times more chances
to get ``13 13 13 12 12 12`` when compared to ``16 15 13 8 8 8``

This tables suggest next scenario to generate stats:

1. Roll (2d10 keep largest 1).
2. Check the tables and pick one matching your result.
3. Table will specify what additional dices should be rolled: roll them
4. Pick stats from the line that matches previous result.

By using (2d10 keep largest 1) we can get 10 subtables:
* first table have 1% chance to be used
* last table have 19% chance to be used



Alternative method that is not implemented yet:

1. roll Xd100
2. calculate the sum
3. use tables where low probability stats are specified for single value on dices, and high probability stats are specified for range of dice values