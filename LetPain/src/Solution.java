import org.junit.Test;

import java.util.*;

public class Solution {
    public static void main(String[] args){
        Solution sol = new Solution();
        int[][] intervals = {{3,5},{12,15}};
        int[][] insert = sol.insert(intervals,new int[]{6,6});
        List<int[]> a = new ArrayList<>(Arrays.asList(insert));
        a.forEach(e->System.out.println(e[0] + " " + e[1]));
    }
    public static void listForeach(ListNode listNode){
        while (listNode!=null) {
            System.out.print(listNode.val + " ");
            listNode = listNode.next;
        }
    }
    public static ListNode createList(int[] nums){
        ListNode head = new ListNode(nums[0]);
        ListNode ret = head;
        for(int i=1;i<nums.length;i++){
            ret.next = new ListNode(nums[i]);
            ret = ret.next;
        }
        return head;
    }
    public ListNode addTwoNumbersII(ListNode l1, ListNode l2) {
        ListNode ret = new ListNode();
        ListNode head = ret;
        int carry = 0;
        while(l1!=null && l2!=null){
            int e1 = l1.val,e2= l2.val;
            int sum = e1+e2 +carry;
            carry = sum>=10?1:0;
            head.next = new ListNode(sum %10);
            head = head.next;
            l1 = l1.next;
            l2=l2.next;
        }
        ListNode list = l1!=null?l1:l2!=null?l2:null;
        if(list==null) {
            if (carry == 1) {
                head.next = new ListNode(1);
                head = head.next;
            }
            head.next = null;
            return ret.next;
        }
        while (list!=null){
            int e =list.val;
            int sum = e+carry;
            carry = sum>=10?1:0;
            head.next = new ListNode(sum %10);
            head = head.next;
            list = list.next;
        }
        if (carry == 1) {
                head.next = new ListNode(1);
                head = head.next;
            }
        head.next = null;
        return ret.next;
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        return 0;
    }
    public double getKElement(int[] nums1,int start1,int end1,
    int[] nums2,int start2,int end2,int k,boolean flag){
        if(k==1) {
            if(!flag)
            return Math.min(nums1[start1], nums2[start2]);
            if(nums1[start1]<=nums1[start1])
            {
                int e2= start1+1>=end1?nums2[start2]:Math.min(nums1[start1+1],nums2[start2]);
                return (e2+nums1[start1]) / 2.;
            }else{
                int e2 = start2+1>=end2?nums1[start1]:Math.min(nums1[start1],nums2[start2+1]);
                return (e2+nums2[start2]) / 2.;
            }
        }
        int e1,e2,det1=0,det2=0;
        if(start1==end1)
            return nums2[start2+k];
        if(start2==end2)
            return nums1[start1+k];
        if (start1+k/2-1>=end1)
        {
            e1 = nums1[end1-1];
            det1 = end1-start1;
        }else{
            e1 = nums1[start1+k/2-1];
            det1=k/2;
        }
        if (start2+k/2-1>=end2)
        {
            e2 = nums2[end2-1];
            det2 = end2-start2;
        }else{
            e2 = nums2[start2+k/2-1];
            det2=k/2;
        }
        if(e1<=e2){
            start1 +=det1;
            k =k-det1;
        }else{
            start2 +=det2;
            k = k-det2;
        }
        return getKElement(nums1,start1,end1,nums2,start2,end2,k,flag);
    }
    public int findIndex(int target,int[] nums){
        if(target < nums[0])
            return 0;
        if(target >= nums[1])
            return nums.length;
        int start = 0,end = nums.length,mid = (start +end) /2;
        while(end-start>1){
            if(nums[mid]>target){
                end = mid;
                mid = start +(end-start) /2;
            }else if(nums[mid]<target){
                start = mid;
                mid = start +(end-start) /2;
            }else{
                return mid;
            }
        }
        return start+1;
    }
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> ret = new ArrayList<>();
        Stack<int[]> stack = new Stack<>();
        int left = newInterval[0],right = newInterval[1];
        int start = -1,end = -1;
        int i=0;
        if(newInterval[1]<intervals[0][0])
            ret.add(newInterval);
        for(;i<intervals.length;i++){
            int[] e = intervals[i];

            if(e[0]<= right && e[0]> left || e[1]<=right && e[1]>=left){
                if(start==-1)
                    start = Math.min(left,e[0]);
                end = Math.max(end,Math.max(right,e[1]));
                if(i==intervals.length-1)
                    ret.add(new int[]{start, end});
                continue;
            }

            if(start!=-1 && end !=-1) {
                ret.add(new int[]{start, end});
                start=-1;
            }
            ret.add(e);
            if(i<intervals.length-1 && e[1]<newInterval[0] && intervals[i+1][0]>newInterval[1])
                ret.add(newInterval);
        }
        if(newInterval[0]>intervals[intervals.length-1][1])
            ret.add(newInterval);
        int[][] retArray = new int[ret.size()][];
        for(int j=0;j<ret.size();j++)
            retArray[j] = ret.get(j);
        return retArray;
    }

}
