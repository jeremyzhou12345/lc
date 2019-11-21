
Key:reduce of search space

Template 2 has no need for post-processing


/***********************************************************************/
Search Space is range(smallest to biggest, unsorted array):


mid = smallest + largest/2
count = number of elements that are smaller than mid
if(count > mid)
else if(count < mid)
else...

/**Find the duplicate Number*/ 好题，binary search可不同写法，不同loop termination condition
search space is range.count the number in the array which  are less than the mid. 
s1: [l,r]
s2:[l,r)
	

/**Search in rotated sorted Array*/
1) everytime check if targe == nums[mid], if so, we find it.
2) otherwise, we check if the first half is in order (i.e. nums[left]<=nums[mid]) 
  and if so, go to step 3), otherwise, the second half is in order,   go to step 4)
3) check if target in the range of [left, mid-1] (i.e. nums[left]<=target < nums[mid]), if so, do search in the first half, i.e. right = mid-1; otherwise, target can't exist in the first half. search in the second half left = mid+1;
4)  check if target in the range of [mid+1, right] (i.e. nums[mid]<target <= nums[right]), if so, do search in the second half, i.e. left = mid+1; otherwise search in the first half right = mid-1;

5 6 9 1 3 4
case1: nums[mid] == target
case2: nums[mid] > nums[l]
	case2.1: nums[l] <= target < nums[mid] (target 在l与mid中间，包含l)
	case2.2: target在mid右侧
case2: nums[mid] < nums[l]

易错点: nums[l] can be equal to nums[mid]; and target can be equal the nums[l] or nums[r]

//s1: termination is l+1<r, template 1
    public int search(int[] nums, int target) {
        if(nums == null || nums.length == 0){
            return -1;
        }
        
        int l = 0;
        int r = nums.length - 1;
        while(l+1<r){
            int mid = l + (r - l)/2;
            if(nums[mid] == target)
                return mid;
            else if(nums[mid] > nums[l]){
                if(nums[l] <= target && target < nums[mid])
                    r = mid;
                else
                    l = mid;
            }
            else{
                if(target > nums[mid] && target <= nums[r])
                    l = mid;
                else
                    r = mid;
            }
        }
        
        if(nums[l] == target) return l;
        if(nums[r] == target) return r;
        return -1;
        
    }

//s2: termination condition is l<=r
    public int search(int[] nums, int target) {
        if(nums == null || nums.length == 0){
            return -1;
        }
        int l = 0;
        int r = nums.length - 1;
        while(l<=r){
            int mid = l + (r-l)/2;
            if(target == nums[mid])
                return mid;
            if(nums[l] <= nums[mid]){   //一定是<=,易错点
                if(nums[l] <= target && target < nums[mid])
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            else{
                if(nums[mid] < target && target <= nums[r])
                    l = mid + 1;
                else
                    r = mid - 1;
            }
        }
        return -1;
    }

/**  Find First and Last Position of Element in Sorted Array */
5 7 7 8 8 10
first find the most left, then most right, then return

    public int[] searchRange(int[] nums, int target) {
        int[] res = {-1, -1};
        if(nums == null || nums.length == 0)
            return res;
		//find the most left, may not exist
		int l = 0;
		int r = nums.length - 1;
		while(l+1<r){
			int mid = l + (r - l)/2;
			if(nums[mid] >= target)
				r = mid;
			else
				l = mid;
		}
		if(nums[l] == target)
			res[0] = l;
		else if(nums[r] == target)
			res[0] = r;
		else
			return res;
		
		//find the most left, may not exist
		int l = 0;
		int r = nums.length-1;
		while(l+1 < r){
			int mid = l+(r-l)/2;
			if(nums[mid] <= target)
				l = mid;
			else
				r = mid;
		}
		if(nums[r] == target)
			res[1] = r;
		else if(nums[l] == target)
			res[1] = l;
		else
			return res;
		
		return res;
		
        
    }
\

//binary search, template 2


/**Kth Smallest Element in a sorted Matrix*/
//When the array is unsorted, and to find a specific number, search space is range
duplicated element exists

subproblem: how to find the largest element which is smaller or equal to the target in sorted matrix
Time Complexity: O(N) for finding the target. Therefore log(max-min)*O(N) for Binary Search
int j = matrix[0].length - 1;
int count = 0;
从最右上角开始
for(int i = 0; i < matrix.length; i++){
	while(j>=0&&matrix[i][j]>target) j--;
	count+=(j+1);
}
if(count<k) l = mid + 1;
else		r = mid;
 
edge case: [l,r]= [2,3]
    public int kthSmallest(int[][] matrix, int k) {
        //corner case
		//if(matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
		int l = matrix[0][0];
		int r = matrix[matrix.length - 1][matrix.length-1];	//[l,r]
		int n = matrix.length;
		while(l<r){
			int mid = l+(r-l)/2;
			int j = matrix[0].length - 1;
			int count = 0;		//count is number of element smaller or equal than target
			for(int i = 0; i < matrix.length; i++){
				while(j>=0&&matrix[i][j]>target) j--;
				count+=(j+1);
			}
			if(count > k)
				r = mid;    //there might be lots of mid. Therefore r might be the kth smallest
            else if(count == k)
                r = mid;
			else
				l = mid + 1;
			
		}
		return l;
    }

//s2: Heap






/**Find minimum in Rotated Sorted Array*/
//the problem is that the array might not be rotated. It could be [1,2,3,4,5]. Compare with the most right element to avoid this problem
//All the element to the right of inflection point < first element; while all elements to the left of inflection point >= first element

    public int findMin(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while(l<r){
            int mid = l + (r-l)/2;
            if(nums[mid] > nums[r])
                l = mid + 1;	//the mini can't be nums[mid], therefore mid+1
            else 
                r = mid;		//the minimum can be nums[mid]
        }
        return nums[l];
    }
	
//follow up: there exists duplicated elements
//corner case: pivot at duplicated element, for example, [3,3,3,3,1,2,3], and the nums[mid] == nums[right]
//When nums[mid] == nums[l], we don't know whether mid is on the left part or right part. 
    public int findMin(int[] nums) {
        int l = 0;
        int r = nums.length - 1;    //[l,r]
        
        while(l<r){
            int mid = l + (r-l)/2;
            if(nums[mid] > nums[r])
                l = mid + 1;
            else if(nums[mid] == nums[r])
                r--;    
            else    //nums[mid] < nums[r]
                r = mid;
        }
        return nums[r];
    }

	
/**Find Peak Element*/
//corner case: all elements ascending: peak is the last element; 
//corner case: all elements descending: peak is the first element 
//use binary search to find boundary in order to reduce search space, [l,r], which ensures the peak among [l,r]
//if middle lies on a rising slope, peak must be within [mid+1, r]
//if middle lies on a falling slope, peak must be within [l, mid-1]

    public int findPeakElement(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while(l<r){
            int mid = l+(r-l)/2;
            if(nums[mid] < nums[mid+1])
                l = mid+1;
            else if(nums[mid] > nums[mid+1])
                r = mid;
        }
        return l;
    }
	

/**Search in Rotated Sorted Array 2*/
//follow up: there might exist duplicated elements
Difference with "no duplicated element" version is:
When nums[l] == mid, [l,mid] might be in order, or out of order,[3,4,1,2,3,3,3,3,3]. It is garanteed that in this 
case, nums[mid] == nums[r] as well. Therefore, check if nums[mid] == nums[l] == nums[r] at first, if so, move left
pointer towards the middle by 1.

    public boolean search(int[] nums, int target) {
        if(nums == null || nums.length == 0)
            return false;
        int l = 0;
        int r = nums.length - 1;
        while(l<=r){
            int mid = l+(r-l)/2;
            if(nums[mid] == target)
                return true;
            if(nums[mid] == nums[l] && nums[mid] == nums[r]){
                l++;
                continue;
            }
            if(nums[mid] >= nums[l]){
                if(nums[l] <= target && target < nums[mid])
                    r = mid - 1;
                else
                    //rule out [l,mid]
                    l = mid + 1;
            }
            else{
                if(nums[mid] < target && target <= nums[r])
                    l = mid + 1;
                else
                    r = mid - 1;
            }
        }
        return false;

/**Median of Two Sorted Arrays*/

since i is ranging within [0,m], only n >= m can ensure j is always valid no matter where i lies on.(j >=0 and j <= n)
check return at the end of the program

       public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
		int n = nums2.length;
		//to ensure n >= m
		if(m > n){
			int[] temp = nums1;
			nums1 = nums2;
			nums2 = temp;
			m = nums1.length;
			n = nums2.length;
		}
		
		
		int iMin = 0, iMax = m, halfLen = (m+n+1)/2;
		while(iMin<=iMax){
			int i = iMin+(iMax-iMin)/2;	//i is the index in the right set
			int j = halfLen - i; 	//index
			
			if(j > 0 && i < m && nums2[j-1] > nums1[i]){
				iMin = i+1;
			}
			else if(i > 0 && j < n && nums1[i-1] > nums2[j]){
				iMax = i-1;
			}
			else{
				//i is perfect
				int maxLeft = 0;
				if(i == 0)
					maxLeft = nums2[j-1];
				else if(j == 0)
					maxLeft = nums1[i-1];
				else{
					maxLeft = Math.max(nums1[i-1],nums2[j-1]);
				}
				if((m+n)%2 == 1)
					return maxLeft;
				
				int minRight = 0;
				if(i == m)
					minRight = nums2[j];
				else if(j == n)
					minRight = nums1[i];
				else
					minRight = Math.min(nums1[i], nums2[j]);
				
				return (maxLeft + minRight)/2.0;
			}
		}
           
        return 0.0;
		
		
    }

/**Missing number*/
s1: bit Manipulation
全部异或
s2: Binary search in range
result must not be ruled out: l = mid + 1; r = mid;
loop termination不能死循环, when there is only two or single element in the array for checking 
这道题因为不用检查l==r的情况，所以不会死循环；
如果需要检查l==r(当只有一个元素的情况),或者指针移动条件为l=mid; r = mid - 1(只有两个元素的情况)，就会死循环
    public int missingNumber(int[] nums) {
        Arrays.sort(nums);
        int l = 0; int r = nums.length;
        while(l < r){
            int mid = l + (r-l)/2;
            if(nums[mid] == mid)
                l  = mid + 1;
            else if(nums[mid] > mid)
                r = mid;
        }
        return l;
    }
或者就用第一个模板，最后做post-processing,也不会rule out或者死循环

