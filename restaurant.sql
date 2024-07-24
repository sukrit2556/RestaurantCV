-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jul 25, 2024 at 12:34 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `restaurant`
--

-- --------------------------------------------------------

--
-- Table structure for table `customer_events`
--

CREATE TABLE `customer_events` (
  `customer_ID` int(8) NOT NULL,
  `tableID` int(2) NOT NULL,
  `customer_amount` int(3) NOT NULL,
  `customer_IN` datetime NOT NULL,
  `customer_OUT` datetime NOT NULL,
  `time_getFood` datetime NOT NULL,
  `captured_video` varchar(200) NOT NULL,
  `getfood_frame` varchar(200) NOT NULL,
  `created_datetime` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `employee`
--

CREATE TABLE `employee` (
  `employee_ID` int(8) NOT NULL,
  `employee_name` varchar(250) NOT NULL,
  `employee_sname` varchar(250) NOT NULL,
  `employee_image` varchar(250) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `suspicious_events`
--

CREATE TABLE `suspicious_events` (
  `sus_ID` int(10) NOT NULL,
  `sus_type` int(2) NOT NULL COMMENT '0 = drawer, 1 = employee',
  `sus_employeeID` int(8) DEFAULT NULL,
  `sus_video` varchar(250) NOT NULL,
  `sus_status` int(2) NOT NULL COMMENT '0 = resolved, 1 = not resolved',
  `sus_datetime` datetime NOT NULL,
  `sus_where` int(2) NOT NULL COMMENT '0 = cashier, 1-6 = table 1-6'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `table_list`
--

CREATE TABLE `table_list` (
  `tableID` int(2) NOT NULL,
  `table_status` int(1) NOT NULL COMMENT '0 = unoccupied\r\n1 = occupied\r\n2 = unoccupied dirty'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `customer_events`
--
ALTER TABLE `customer_events`
  ADD PRIMARY KEY (`customer_ID`,`created_datetime`);

--
-- Indexes for table `employee`
--
ALTER TABLE `employee`
  ADD PRIMARY KEY (`employee_ID`);

--
-- Indexes for table `suspicious_events`
--
ALTER TABLE `suspicious_events`
  ADD PRIMARY KEY (`sus_ID`,`sus_datetime`);

--
-- Indexes for table `table_list`
--
ALTER TABLE `table_list`
  ADD PRIMARY KEY (`tableID`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `customer_events`
--
ALTER TABLE `customer_events`
  MODIFY `customer_ID` int(8) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `employee`
--
ALTER TABLE `employee`
  MODIFY `employee_ID` int(8) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `suspicious_events`
--
ALTER TABLE `suspicious_events`
  MODIFY `sus_ID` int(10) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
