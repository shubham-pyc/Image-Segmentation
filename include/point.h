struct Point
{
    int x{0};

    inline bool operator==(Point a) const
    {
        return a.x == x;
    }
};
