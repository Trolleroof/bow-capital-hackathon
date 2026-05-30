# ORB-SLAM3 Source Patches

## 1. LocalInertialBA assertion fix

**File:** `src/Optimizer.cc`, line 2837

**Problem:** Hard `assert(mit->second>=3)` crashes the process when a keyframe in the local BA window has fewer than 3 map point observations. This is a known ORB-SLAM3 bug that fires under normal stereo-inertial operation.

**Original:**
```cpp
for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
{
    assert(mit->second>=3);
}
```

**Patched:**
```cpp
for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
{
    if(mit->second<3)
        return;
}
```

**Effect:** Skips the local inertial BA optimization for that round instead of aborting. Tracking continues normally.
