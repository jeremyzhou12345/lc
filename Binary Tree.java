
总结方法：每道题只做一遍，
要有自信如果出现原题或者类似题能在规定时间做出并描述出来
用recursion解决问题一定想清楚function signature
Consider: do I need to make sure its children is not null before next step(recursion)? case1: all null; case2:one of child is null; case3: all not null

对于满二叉树，可以通过子index找到parent index, 反之也成立


/**max depth of binary tree*/
//solution 1:
public int maxDepth(TreeNode root){
	if(root == null)
		return 0;
	int left = maxDepth(root.left);
	int right = maxDepth(root.right);
	return Math.max(left,right)+1;
}

//solution 2: traversal (from bottom to top)
int globalMax = 0;
public int maxDepth(TreeNode root){
	if(root == null)
		return 0;
	
	helper(root, 1);
	return globalMax;
}

private void helper(TreeNode root, int depth){
	if(root == null){
		if((depth-1) > globalMax)
			globalMax = (depth-1);
		return;
	}
	
	helper(root.left, depth+1);
	helper(root.right, depth+1);
}
/**Minimum Depth of Tree*/
s1:global variable and update at base case(叶子节点). Pass height of tree to children tree nodes
s2: from bottom to up. find height from left and right children. 注意分类讨论
s3: iteratively traverse the tree with current height. update minimum depth when the current node is leaf 
s4: BFS for minimum path problem.iteratively traverse the tree by level, the first leaf we reach is minDepth.no
need to iterate all nodes

/**balanced binary tree*/


/**lowest common ancestor*/
碰到，返回
root不是，左右子树分情况讨论

/**lowest common ancestor for BST*/
利用性质
case1: two nodes values larger than current root node
case2: two nodes values smaller than current root node
case3: else: current node is common ancestor

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == null || q == null)
            return null;
        if(root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p ,q);
        else if(root.val < p.val && root.val <q.val)
            return lowestCommonAncestor(root.right, p ,q);
        else
            return root;
    }
//follow up: with parent node: 从p,q分别建造到root的路径//找到length diff, 同时向上走


/**lowest common ancestor 2 with parent nodes*/
If both the nodes p and q are in the right subtree, then continue the search with right subtree starting step 1.
If both the nodes p and q are in the left subtree, then continue the search with left subtree starting step 1.
If both step 2 and step 3 are not true, this means we have found the node which is common to node p's and q's subtrees. 
and hence we return this common node as the LCA.



/**BST iterator*/
Space complextiy: O(h) where h is the height of tree

define a function to: store current root's all left children into stack

First store all root's left children into stack. 
After poppoing current min TreeNode, when the min TreeNode has a right branch, use function above to store its right branch's all left child nodes

class BSTIterator {


    Deque<TreeNode> stack = new LinkedList<>();
    public BSTIterator(TreeNode root) {
        storeAllLeftChild(root);
    }
    
    private void storeAllLeftChild(TreeNode root){
        while(root != null){
            stack.offerFirst(root);
            root = root.left;
        }
    }
    
    /** @return the next smallest number */
    public int next() {
        TreeNode cur = stack.pollFirst();
        storeAllLeftChild(cur.right);
        return cur.val;
    }
    
    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        if(stack.isEmpty())
            return false;
        return true;
    }
}



/**Minimum Depth of Binary Tree*/


    public int minDepth(TreeNode root) {
        //base case
        if(root == null)
            return 0;
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        
        //case1: root is leaf:
        if(left == 0 && right == 0)
            return left+right+1;
        //case2:one child
        if(left == 0 || right == 0)
            return left+right+1;
        else
            return Math.min(left,right)+1;
    }
	
	
	
/**Sum root to Leaf Number*/

Input: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
//s1
    public int sumNumbers(TreeNode root) {
        int[] sum = new int[]{0};
        if(root != null)
            helper(root,0,sum);
        return sum[0];
    }
    
    private void helper(TreeNode root, int val, int[] sum){
        val += root.val;
        
        //base case
        if(root.left == null && root.right == null){
            sum[0]+=val;
            return;
        }
        
        if(root.left != null)
            helper(root.left, val*10, sum);
        if(root.right!=null)
            helper(root.right,val*10,sum);
    }
	
//s2:
    4
   / \
  9   0
 / \
5   1
通过参数向下传：49(current sum);向上传：该分支的final result:495
在当前层做什么：检查是否为Null,检查是否为leaf.否则向下传递当前sum
    public int sumNumbers(TreeNode root) {
        return sumHelper(root,0);
    }
    
    private int sumHelper(TreeNode root, int sum){
        if(root == null)
            return 0;
        if(root.left == null && root.right == null){
            return sum*10+root.val;
        }
        int leftRes = sumHelper(root.left,sum*10 + root.val);
        int rightRes = sumHelper(root.right,sum*10+root.val);
        return leftRes+rightRes;
    }
	
	
	
/**invert binary tree*/
//recursion:
function of "invertTree": invert a binary tree from TreeNode "root"
    public TreeNode invertTree(TreeNode root) {
        if(root == null)
            return null;
        TreeNode temp = root.right;
        root.right = invertTree(root.left);
        root.left = invertTree(temp);
        return root;
    }

/**Tweaked Identical Binary Trees*/
Time Complexity: O(n^2)

  public boolean isTweakedIdentical(TreeNode one, TreeNode two) {
    // Write your solution here
    if(one == null && two == null)
		return true;
    else if(one == null)
		return false;
    else if(two == null)
		return false;
    else if(one.key != two.key)
		return false;

    return (isTweakedIdentical(one.left,two.left)&&isTweakedIdentical(one.right,two.right))
    ||(isTweakedIdentical(one.left,two.right)&&isTweakedIdentical(one.right,two.left));
  }
  
  
/**Count Univalue Subtrees*/
从上往下，从下往上
A tree is univalue subtree if:
case1: node is leaf  
case2: both children are univalue tree and the node and both its children have the same value

recursion rule
base case
However, method 1 solution includes lots of repeated calculation。 bottom-up can solve this problem.Therefore,
post-order traversal.

知识点易错点： || short circuits but | does not. 比如&&，只要第一个为false,第二个就不再计算。但我们可能需要计算第二个，即使第一个为false
。比如这道题，即使左子树为false,但任需计算右子树中univalue tree的个数，依然要用recursion(root.right)
s1: recursion one time


s2: pass parent values
mistake: | instead of ||. | won't short circuit
        if(!helper(root.left,root.val)|!helper(root.right,root.val))
            return false;
        else
            res++;


/**Diameter of binary tree*/
Path问题：
问孩子要：该孩子以下的路线信息（这道题：左孩子的最长length）
向上传：包含当前结点的路线信息（如包含该node往下的最长path的length）



/**Distance of Two Node*/
//观察path between two node 的特点：always includes their lowest common ancestor
Therefore, the problem is to find their common ancestor and then calculate distance between ancestor and target node

  public int distance(TreeNode root, int k1, int k2) {
    // Write your solution here
    TreeNode ancestor = LCA(root, k1, k2);
    //distance between ancestor and k1
    int d1 = getDistance(ancestor, k1);
    //distance between ancestor and k2
    int d2 = getDistance(ancestor, k2);
    //result
    return d1+d2;

  }
private int getDistance(TreeNode root, int k){
    if(root == null)
    return -1;
    if(root.key == k)
        return 0;
    int l = getDistance(root.left, k);
    int r = getDistance(root.right, k);

    return Math.max(l,r) == -1 ? -1 : Math.max(l,r)+1;
}


//function signature: return the lowest common ancestor between k1 and k2
  public TreeNode LCA(TreeNode root, int k1, int k2){
    if(root == null)
        return root;
    
    if(root.key == k1 || root.key == k2)
        return root;

    TreeNode left = LCA(root.left, k1, k2);
    TreeNode right = LCA(root.right, k1, k2);

    if(left == null)
    return right;
    else if(right == null)
    return left;
    else
        return root;
  }
  
  /**Binary Tree Path*/
  //s1: recursion
StringBuilder 是在原StringBuilder上append.因此需要用sb.delete(int start, int end)

用完StringBuilder记得答案要求为返回String。Stringbuilder.toString()

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res= new ArrayList<>();
        if(root == null)
            return res;
        StringBuilder sb = new StringBuilder();
        helper(res,root,sb);
        return res;
    }
    
    private void helper(List<String> res, TreeNode root, StringBuilder sb){
        int size = sb.length();
        if(root.left == null && root.right == null){
            res.add(sb.append(root.val).toString());
            sb.delete(size,sb.length());
            return;
        }
        sb.append(root.val);
        sb.append("->");
        
        if(root.left!=null)
            helper(res,root.left, sb);
        if(root.right!=null)
            helper(res,root.right,sb);
        sb.delete(size,sb.length());
        
    }
    }
/**Path sum*/
//recursion 
//iteration

/**Path Sum 2*/


/**Path Sum 3: find the number of path that sum to a given value,直上直下path*/
//s1: recursion:
result = starting from root node的path有几条 + left child node的path有几条 +right child node的path有几条

starting from root node的path有几条 = (current node.val == sum? 1: 0) + starting from left child node, sum为 target sum - root.val，的path有几条 
starting from right node的path有几条 同理

    public int pathSum(TreeNode root, int sum) {
        if (root == null) return 0;
        return pathSumFrom(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
    
    private int pathSumFrom(TreeNode node, int sum) {
        if (node == null) return 0;
        return (node.val == sum ? 1 : 0) 
            + pathSumFrom(node.left, sum - node.val) + pathSumFrom(node.right, sum - node.val);
    }
	
	Time Complexity:
	Space Complexity:
	
//s2: keep a prefix and use hashSet to find sum

	public int pathSum(TreeNode root, int sum){
		if(root == null)
			return 0;
		return helper(root,sum,new HashSet<Integer>(),0);
	}
	
	private int helper(TreeNode root, int sum, Set<Integer> set,int curSum){
		if(root == null)
			return 0;
		curSum = curSum+root.val;
		int temp = 0;
		if(set.contains(curSum-sum))
			temp = 1;
		set.add(curSum);
		int left = helper(root.left, sum, set, curSum);
		int right = helper(root.right,sum,set,curSum);
		set.remove(curSum);
		return left+right+temp;
	}



/**Maximum Path Sum Binary Tree 3*/
/**find the miximum sum from any node to any node(must go downwards, travelling only from parent nodes to child nodes)*/
  //s1: recursion
  问左右孩子要最大值（including左右孩子）
向上传包含current node的最大值
  public int maxPathSum(TreeNode root) {
    // Write your solution here
    if(root == null)
      return 0;
    int[] globalMax = new int[]{Integer.MIN_VALUE};
    helper(root,globalMax);
    return globalMax[0];
  }
  
  private int helper(TreeNode root,int[] globalMax){
      if(root == null)
          return 0;
      int left = Math.max(0,helper(root.left,globalMax));
      int right = Math.max(0,helper(root.right,globalMax));
      int cur = Math.max(left,right)+root.key;
      if(cur > globalMax[0])
        globalMax[0] = cur;
      return cur;
  }
  
  //s2: keep a prefix and find the maximum consecutive sequence in prefix by DP
  public int maxPathSum(TreeNode root){
	  if(root == null)
		  return 0;
	  int[] max = new int[]{Integer.MIN_VALUE};
	  helper(root,new ArrayList<>(),max);
	  return max[0];
  }
  
  private void helper(TreeNode root, List<Integer> prefix,int[] max){
	  if(root == null)
		  return;
	  prefix.add(root.val);
	  if(root.left == null && root.right == null){
		  max[0] = Math.max(findMax(prefix),max[0]);
		  prefix.remove(prefix.size()-1);
		  return;
	  }
	  helper(root.left,prefix,max);
	  helper(root.right,prefix,max);
	  //List prefix 出去的时候和进来前必须一样,therefore don't forget to remove 
	  prefix.remove(prefix.size()-1);
	  return;
  }
  
  private int findMax(List<Integer> prefix){
	  int dp = 0;
	  int max = Integer.MIN_VALUE;
	  for(int i : prefix){
		  if(dp > 0)
			  dp = dp + i;
		  else 
			  dp = i;
		  max = Math.max(max,dp);
	  }
	  return max;
  }
  
  /**Tree serialization problem*/
  /**Serialize and deserialize binary tree*/
树的构造与重建：
To serialize tree, we can traverse the tree, including dfs/bfs. DFS is more straightforward for deserialization,
since the adjacent nodes is naturally encoded in the order
s1: DFS
Se:traversal tree and add tree value into result, including null
De:gloabl pointer indicating what the current node should be converted into tree node.// or queue so current node is queue.poll();
However, it says no class global variable can be used. Therefore, use int[] i = int[]{0} as paremeter of recursion function

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] arr = data.split(",");
		List<String> list = new ArrayList<>();
		for(String s : arr){
			list.add(s);
		}
        return helper(list, new int[1]{0});
    }
    private TreeNode helper(List<String> list, int[] i){
        if(list.get(i[0]).equals("null")){
			i[0]++;
            return null;
        }else{
            TreeNode root = new TreeNode(Integer.parseInt(list.get(i[0]++)));
            root.left=helper(list,i);
            root.right=helper(list,i);
            return root;
        }
    }
	follow-up: what if no regex is allowed
	edge case for converting string to integer: negative number 
	private TreeNode helper(String s, int[] i){
		if(s.charAt(i[0])=='#'){
			i[0]++;
			return null;
		}
		boolean isNegative = false;
		int num = 0;
		if(s.charAt(i[0])=='-'){
			isNegative =true;
			i[0]++;
		}
		while(i[0]!=','){
			num = num*10+(s.charAt(i[0])-'0');
			i[0]++;
		}
		i[0]++;
		if(isNegative)
			num=-num;
		TreeNode root= new TreeNode(num);
		root.left = helper(s,i);
		root.right = helper(s,i);
		return root;
	}

s2:BFS
dese: poll left and right nodes from queue and add to current tree node. If not null, add left and right nodes
into another queue and wait for connecting their children nodes
注意 ArrayDeque不能储存null

/**Serialize and deserialize BST*/

preOrder and postOrder traversal are both suitable for compact serialization.(no "null" needed)

preOrder(or postOrder also works) to serialize the tree. 
Deserialize:
s1: construct root and find left sub nodes and right sub nodes. Both array, which index indicates the start and 
end of subtree, and queue, which contains left/right subtree nodes, can work. Here queue is perferred because 
the base case is easier to handle.

(Important!!)s2: preOrder simulator to construct tree. Use limit of range of left/right subtree to indicate that whether 
the node is left/right subnode or return null.

follow up: optimize space and time by converting int to 4-bytes string. We can also get rid of delimiters.
String "123456" takes 6 bytes. int a = 123456 in binary format is 4*8 bits. store each 8 bits by char, so that
int a can be stored by char[] of size 4. String s = new String(char[]). 
Since each number is a 4 bytes string, no delimiters are required
public String intToString(int x){
	char[] bytes = new char[4];
	for(int i = 3; i>=0; i--){
		bytes[i]=(char)(x&0xff);
		x=x>>8;
	}
	return new String(bytes);
}
public int stringToInt(String s){
	int result= 0;
	for(char c : s.toCharArray()){
		result = (result<<8)+int(c);
	}
	return result;
}

  //covert a tree into double linked list using in-order traversal sequence
  Application of in-order traversal. 
  
  class Solution{
	public TreeNode toDoubleLinkedList(TreeNode root) {
    // Write your solution here.
		TreeNode newHead = null;
		TreeNode prev = null;
		helper(root, prev, newHead);

		return newHead;
  }

  private void helper(TreeNode root, TreeNode prev, TreeNode newHead){
	  if(root == null)
		  return;
	  //把左边child已经连好
	  helper(root.left,prev,newHead);
    if(prev == null){
      newHead = root;
      root.left = prev;
    }
    else{
    prev.right = root;
    root.left = prev;
    }
	  prev = root;
	  helper(root.right,prev,newHead);
	  return;
	  
  }
  
  }
/**serialize and deserialize N-ary Tree*/
we must know how many children each node has. 
Serialized sequence of Example: 1,3,3,2,5,0,6,0,2,0,4,0
mistake: index indicating the current node must be int[] reference instead of int. 
  
/**reconstruct binary tree with inorder and preorder traversal*/
divide and conquer:
function signature:
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder == null || inorder == null){
		    return null;
	    }
	    HashMap<Integer,Integer> lookup = new HashMap<>();
	    for(int i = 0; i < inorder.length;i++){
		    lookup.put(inorder[i],i);
	    }
		//[pre_l,pre_r],[in_l,in_r]
	    TreeNode root = helper(preorder,0,preorder.length-1,inorder,0,inorder.length-1,lookup);
	    return root;
    }

	private TreeNode helper(int[] preorder, int pre_l, int pre_r, int[] inorder, int in_l,int in_r, HashMap<Integer, Integer> lookup){
		if(pre_l > pre_r){
			return null;
		}
		
		TreeNode root = new TreeNode(preorder[pre_l]);
		int indexInOrder = lookup.get(preorder[pre_l]);
		int leftSubTreeLength = indexInOrder-in_l;
		int rightSubTreeLength = in_r-indexInOrder;
		root.left = helper(preorder,pre_l+1,pre_l+leftSubTreeLength,inorder,in_l,indexInOrder-1,lookup);
		root.right = helper(preorder,pre_l+leftSubTreeLength+1,pre_r,inorder,indexInOrder+1,in_r,lookup);
		return root;
	}

  
  
/**construct binary search tree from preorder traversal*/
BST: set bound
体会preorder的特性：for current node, if it has child, if must be the next value after in preorder traversal

class Solution {
    int i = 0;
    public TreeNode bstFromPreorder(int[] preorder) {
        if(preorder == null || preorder.length == 0)
            return null;
        return bstHelper(preorder,Integer.MAX_VALUE);
    }
    
    private TreeNode bstHelper(int[] preorder, int bound){
        if(i == preorder.length||preorder[i] > bound){
            return null;
        }
        
        TreeNode root = new TreeNode(preorder[i++]);
        
        root.left = bstHelper(preorder,root.val);
        root.right = bstHelper(preorder,bound);
        
        return root;
    }
}
  
  
 /**convert sorted list to binary search tree*/
 //s1: divide and conquer
 
 recursion rule: find middle node between head and tail, generate left subtree with [head,middle), root is middle
 ,generate right subtree with [middle.next,tail).
 case1: middle is the same as head: Required: both left and right subtree return null, root is middle node 
 case2: middle.next is tail, right subtree return null

 base case: if head==tail, return null 
 //Time Complexity: each time find middle cost 2/N, logN level, therefore O(NlogN)
 //Space: since balanced tree, Space is height of recursion stack, logN
    public TreeNode sortedListToBST(ListNode head) {
        //edge case: null or single node
        return toBST(head,null);    //no generation of tail node
    }
    private TreeNode toBST(ListNode head, ListNode tail){
        if(head==tail)
            return null;
        ListNode slow = head;
        ListNode fast = head;
        while(fast!=tail&&fast.next!=tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = toBST(head,slow);
        root.right = toBST(slow.next,tail);
        return root;
    }
//s2: inOrder traversal
Time Complexity: O(N). Space: logN
Sorted List has the same order with in-order traversal. We know that the left-most node is the first listNode, 
and the next element in the BST will also be the second list node.
We need to tell which nodes belongs to current node's subtree to make balanced tree. Therefore use start/end 
class Solution {
    ListNode head;		//global pointer
    private int findSize(ListNode head){
        int c = 0;
        while (head != null) {
        head = head.next;  
        c += 1;
    }
    return c;
    }
	//convert nodes in the list between start and end to BST
    private TreeNode convertListToBST(int start, int end){
        if(start > end)
            return null;
        int mid = start+(end-start)/2;
        TreeNode left = convertListToBST(start,mid-1);
        TreeNode root = new TreeNode(this.head.val);
        root.left = left;
        this.head = this.head.next;
        root.right = convertListToBST(mid+1,end);
        return root;
    }
    public TreeNode sortedListToBST(ListNode head) {
        int size = findSize(head);
        this.head = head;
        return convertListToBST(0,size-1);
    }

/**114 flatten binary tree to linked list*/
//s1: iteration by pre-order traversal. With the helper of previous node reference.
//s2: recursion. for a node, must ensure all its children all correctly connected before change the node's
reference. Therefore use post-order, start from the last node(most right node)
  
  
/**Binary Tree Upside Down*/
All right nodes are leaf nodes with sibling: for parent node, as long as it has right child node, it always has left
child node. And as long as it has right child node, it must be a leaf

//s1: recursive
父子关系型:recursion调用时同时记录（传参）cur and parent(or child)

不停的对左结点递归直到最左结点，开始翻转，翻转好最左结点后，开始回到上一个左子节点继续翻转直到顶点也被翻转了
public TreeNode UpsideDownBinaryTree(TreeNode root){
	helper(root,null);
}
private TreeNode helper(TreeNode root,TreeNode parent){
	if(root == null) return parent;

	TreeNode newRoot = helper(root.left,root);
	root.left = parent==null?null:parent.right;
	root.right = parent;
	return newRoot;
}

Or not using parent node(more complicated)
To convert:
newHead = root.left;
newHead.left = root.right;
newHead.right = root;

public TreeNode UpsideDownBinaryTree(TreeNode root){
	return helper(root);
}
private TreeNode helper(TreeNode root){
	//base case
	if(root == null)
		return root;
	if(root.left == null){
		new TreeNode newRoot = root;
		return newRoot;
	}
	TreeNode newRoot = helper(root.left);
	TreeNode newHead = root.left;
	newHead.left = root.right;
	newHead.right = root;
	//I made mistake as I didn't consider root is still connected 
	root.left = null;
	root.right = null;
	return newRoot;
}

//s2: iterative solution
不会
when new child nodes is connected, lost the original child nodes' reference ---- solution: store the original 
child nodes'reference before reverse the tree.
This code is very similar to the algorithm in reversing Linked List
public TreeNode UpsideDownBinaryTree(TreeNode root){
	TreeNode cur = root;
	TreeNode parent = null;
	TreeNode leftChild =null;
	TreeNode parentRight=null;
	while(cur!=null){
		leftChild = cur.left;
		cur.left = parentRight;
		parentRight = cur.right;
		cur.right = parent;
		parent = cur;
		cur=leftChild;
	}
	
	return parent;
	
}

/**Find Bottom Left Tree Value*/
//s1: keep a max depth and update the value when depth > max_Depth.
//pre-order traverse ensures the most left node is the first one visited on the same row 
class Solution {
    int maxDepth = 0;
    int res = 0;
    public int findBottomLeftValue(TreeNode root) {
        helper(root,1);
        return res;
    }
    private void helper(TreeNode root, int depth){
        if(root == null) return;
            if(depth > maxDepth){
                res = root.val;
                maxDepth= depth;
            }
        helper(root.left,depth+1);
        helper(root.right,depth+1);
    }
}
//s2: BFS
    public int findBottomLeftValue(TreeNode root) {
        int res = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0;i<size;i++){
                TreeNode temp = queue.poll();
                if(i == 0)
                    res = temp.val;
                if(temp.left!=null)
                    queue.offer(temp.left);
                if(temp.right!=null)
                    queue.offer(temp.right);
            }
        }
        return res;
    }

/**count complete tree nodes*/

height of complete tree can be found by going left.  

思路是由 root 根结点往下，分别找最靠左边和最靠右边的路径长度，如果长度相等，则证明二叉树最后一层节点是满的，
是满二叉树，直接返回节点个数，如果不相等，则节点个数为左子树的节点个数加上右子树的节点个数再加1(根节点)，
其中左右子树节点个数的计算可以使用递归来计算

Time Complexity: O(h^2), h is height of tree
    public int countNodes(TreeNode root) {
        if(root==null)
            return 0;
        int leftH = 0;
        int rightH = 0;
        TreeNode leftP = root;
        TreeNode rightP = root;
        while(leftP!=null){
            leftP = leftP.left;
            leftH++;
        }
        while(rightP!=null){
            rightP = rightP.right;
            rightH++;
        }
        if(leftH==rightH) return (int)Math.pow(2,leftH)-1;
        return countNodes(root.left)+countNodes(root.right)+1;
    }
	
    public int countNodes(TreeNode root) {
        if(root == null)
            return 0;
        int l = height(root.left);
        int r = height(root.right);
        return (l==r)?1<<l+countNodes(root.right):1<<r+countNodes(root.left);
    }
    
    private int height(TreeNode root){
        if(root==null){
            return 0;
        }
        return height(root.left)+1;
    }  
  
/**Find the node with the max difference in the total number of descendents in its left and right subtree*/
从下往上反值：
what to expext from children: number of nodes in left subtree and right subtree
what to do in the current level: compare the difference
what to report to parent: number of nodes of current node

/**Binary Tree Coloring Game*/

trick: 如何同时获得target Node左右子树的node总数: since return can only return one object, 
不用recursion传递，use a global variable 
尽量一遍recursion解决问题
l,r is gloabl variable.
if(root.val == target){
	l = countNodes(root.left);
	r = countNodes(root.right);
}

/**Bianry Tree Path**/
method: dfs(both recursion and iteration can work) or bfs. Stack/Queue always need to store extra information.
Can defind a new Class, or use two Stack/Queue
Can defind a new Class, or use two Stack/Queue
String will store many unused/intermediate string in the heap. Use StringBuilder help this problem
错误的点：To determine how length of string in StringBuilder should be deleted when回溯, we can store the 
length of StringBuilder at the beginning and use StringBuilder.delete(length).

/**Binary Tree Right Side View */
分类： 树的遍历
cur.right->cur.left->cur traversal order ensure that the most right of each level is firstly visited.

/**Populating Next Right Pointers in Each Node*/
base case是tree node with its children
s1: recursion:最小单元。遍历顺序pre-order
s2: iteration. 层序遍历顺序。An Extra node pointer, levelStart is needed 

/**Inorder Successor in BST*/
比较root与p的大小， if root is smaller or equal to p, result must be in the right subtree if right subtree
is not null. If root is larger than p, result could be in left subtree, or current node.
function signature: return smallest key greater than p.val, or null.
public TreeNode successor(TreeNode root, TreeNode p) {
  if (root == null)
    return null;

  if (root.val <= p.val) {
    return successor(root.right, p);
  } else {
    TreeNode left = successor(root.left, p);
    return (left != null) ? left : root;
  }
}
Similarly, largest key smaller than p.val: 
if(root.val > p.val){
	return predecessor(root.left,p);
}else{
	TreeNode right = predecessor(root.right,p);
	return right==null?root:right;
}

/**All Nodes Distance K in Binary Tree*/
//s1: 把问题转化为做过的问题。如果把tree构造成一个图（无向图），就可以用BFS找到答案。
如何构造一个无向图：HashMap
//s2: recursion

/**Binary Tree Vertical Order Traversal*/
//BFS ensure top to bottom and left to right.









